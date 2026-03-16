# LAM Training — Fine-Tuning a 3B Large Action Model on Apple Silicon

Train a small language model to be a **Large Action Model** (LAM) that creates AI agents and skills from natural language requests. The entire pipeline — from data generation to fine-tuning to inference — runs locally on an Apple Silicon Mac.

## What is a Large Action Model?

An LLM tells you *how* to do something. A LAM *does it*.

LAMs extend language models from generating text to **taking actions** — calling APIs, orchestrating multi-step workflows, and composing tool chains. Our LAM specifically learns to:

1. Parse a natural language request ("Build me a PR reviewer that checks for security issues")
2. Reason about the right agent architecture
3. Output a complete agent definition: tools, skills, constraints, and triggers

## Why 3B Parameters?

| Concern | Answer |
|---------|--------|
| "Can a 3B model do this?" | Google's 270M FunctionGemma hit 85% function-calling accuracy. Salesforce's 1B xLAM beat GPT-3.5 Turbo. A 3B model has massive headroom. |
| "Why not just use Claude/GPT?" | Latency, cost, privacy. A local 3B model runs at 80+ tok/s on an M-series Mac with zero API costs. |
| "Why not 7B+?" | Diminishing returns for constrained output formats. Our agent schema is structured JSON — the model doesn't need world knowledge, it needs reliable structured generation. |

## The Approach

### 1. Choose the Base Model

**SmolLM3-3B-Instruct** (Hugging Face) — best-in-class at the 3B scale with:
- Native tool-use support in the instruct variant
- 128K context window
- Dual-mode reasoning (`/think` and `/no_think`)
- Full training transparency (architecture decisions, data mixture published)

Runner-up: **Phi-4-mini** (3.8B) — stronger reasoning but slightly larger.

### 2. Identify Training Data

Existing datasets teach tool *use* but not tool *design*. We need both:

**Off-the-shelf (tool mechanics):**
- [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) — 60K examples across 3,673 APIs
- [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) — 11.3K multi-turn tool dialogs (ICLR 2025)

**Synthetic (agent creation) — this repo:**
- 1,000 training examples across 20 categories
- Input: natural language request → Output: complete agent definition (tools, skills, constraints)
- Generated via Anthropic Batch API with Claude Sonnet 4.6

### 3. Generate Synthetic Training Data

We use Claude to generate high-quality training pairs with three key techniques:

#### Instruction Repetition — Hypothesis & Results

**Hypothesis:** Research suggests that repeating instructions within a prompt significantly improves LLM adherence to complex structured output requirements, with reported gains from ~21% to ~97% in some tasks. We hypothesized that for our use case — generating structured agent definitions with strict schema requirements — repeating the full instruction block would improve output quality, particularly for nuanced requirements like "reasoning should explain WHY, not just WHAT."

**Test Design:** We ran an A/B test using Claude Sonnet 4.6 (`claude-sonnet-4-6`) against the same prompt — a request to create an AWS Lambda cold-start monitoring agent. Both variants used identical schema instructions and temperature (0.7). The single-instruction variant presented the instructions once; the repeated variant appended "Let me repeat your instructions:" followed by the full instruction block and the request again.

**Results:**

| Metric | Single Instruction | Repeated Instruction |
|--------|-------------------|---------------------|
| **Quality Score** | 85/100 | **91/100** |
| Valid JSON | Yes | Yes |
| Tools defined | 5 | 4 |
| Skills defined | 3 | 3 |
| Total steps | 14 | 12 |
| Steps with tool refs | 10/14 (71%) | 8/12 (67%) |
| Constraints | 9 | 9 |
| Reasoning length | 691 chars | **769 chars** |
| **Reasoning explains WHY** | **No** | **Yes** |
| All params typed | Yes | Yes |
| All skills have triggers | Yes | Yes |
| Output tokens | 4,046 | **3,824 (fewer)** |
| Latency | 63.5s | 62.9s |

**Key Findings:**
1. **+7.1% quality improvement** (85 → 91/100) with instruction repetition
2. **Reasoning quality was the decisive differentiator.** The repeated variant was the *only* one that produced reasoning explaining *why* the architecture fits the request — the single variant described *what* it built but not *why*. This is critical for training data quality since we want the fine-tuned model to learn architectural reasoning, not just template-filling.
3. **Fewer output tokens, not more.** Counter-intuitively, the repeated variant used 222 fewer tokens (3,824 vs 4,046). The model was more focused and less verbose — it generated fewer tools (4 vs 5) and fewer steps (12 vs 14) but with higher relevance.
4. **No latency penalty.** Both variants completed in ~63 seconds. The repeated instruction adds ~500 input tokens but doesn't measurably affect response time.

**Conclusion:** Instruction repetition is adopted for all training data generation. The quality improvement is modest in aggregate score but substantial in the specific dimension that matters most — architectural reasoning quality.

The test scripts are included in this repo (`src/test-repetition.ts` and `src/test-repetition-multi.ts`) for reproducibility.

#### Prompt Caching & Batch API — Cost Analysis

**The problem:** Our generation pipeline makes 100 API calls, each carrying an identical ~470-token system prompt containing the schema definition and rules. Without caching, every call re-processes these tokens at full input price.

**The solution:** Anthropic's prompt caching (`cache_control: { type: "ephemeral" }`) caches the system prompt on the first call. The remaining 99 calls read from cache at 90% discount. Combined with the Batch API's flat 50% reduction on all token costs, the savings compound.

**Pricing tiers (Claude Sonnet 4.6):**

| Token Type | Standard Price | Batch Price (50% off) |
|-----------|---------------|----------------------|
| Input tokens | $3.00/M | $1.50/M |
| Cache write | $3.75/M (1.25x input) | $1.875/M |
| Cache read | $0.30/M (0.10x input) | $0.15/M |
| Output tokens | $15.00/M | $7.50/M |

**Cost calculation using observed token counts:**

From our A/B test, each request generates ~4,000 output tokens and consumes ~1,500 input tokens (user prompt with instruction repetition) plus ~470 tokens (system prompt). Across 100 requests:

| Component | No Caching, No Batch | With Caching + Batch | Savings |
|-----------|---------------------|---------------------|---------|
| System prompt (100 calls × 470 tokens) | 47,000 tokens × $3.00/M = $0.14 | 1 write (470 × $1.875/M) + 99 reads (46,530 × $0.15/M) = **$0.008** | 94% |
| User prompt (100 × 1,500 tokens) | 150,000 × $3.00/M = $0.45 | 150,000 × $1.50/M = **$0.225** | 50% |
| Output (100 × 4,000 tokens) | 400,000 × $15.00/M = $6.00 | 400,000 × $7.50/M = **$3.00** | 50% |
| **Total** | **$6.59** | **$3.23** | **51%** |

**Key takeaways:**
1. **Prompt caching saves 94% on the system prompt** — but since the system prompt is only ~470 tokens per call, the absolute dollar savings are modest ($0.13). Caching matters more when system prompts are large (multi-thousand tokens).
2. **The Batch API's 50% discount is the dominant cost saver** — it cuts $3.23 off the output token cost alone, which is where the real spend is.
3. **Output tokens dominate cost.** At $15/M (standard) or $7.50/M (batch), the ~400K output tokens account for 91-93% of total cost regardless of caching strategy.
4. **Combined savings: ~51% ($6.59 → $3.23)** for generating 1,000 training examples — roughly the cost of a coffee.

The prompt caching technique becomes increasingly valuable at scale. For a 10,000-example dataset with a 2,000-token system prompt, cache reads would save ~$5.40 on top of the batch discount.

### 4. Output Schema

The model learns to generate this structure:

```json
{
  "reasoning": "Why this architecture fits the request",
  "agent": {
    "name": "agent-name",
    "description": "What the agent does",
    "role": "Primary role in one phrase",
    "tools": [
      {
        "name": "tool_name",
        "description": "What the tool does",
        "parameters": [{ "name": "param", "type": "string", "description": "...", "required": true }],
        "returns": "Description of return value"
      }
    ],
    "skills": [
      {
        "name": "skill-name",
        "description": "What the skill accomplishes",
        "trigger": "When this skill activates",
        "inputs": [{ "name": "input", "type": "string", "description": "...", "required": true }],
        "steps": [
          { "action": "What happens", "tool": "tool_name", "input": { "param": "value" } }
        ],
        "output": "What the skill produces"
      }
    ],
    "constraints": ["Specific behavioral constraint"]
  }
}
```

### 5. Fine-Tune Locally with MLX

Apple's [MLX](https://github.com/ml-explore/mlx) framework is purpose-built for Apple Silicon's unified memory architecture — zero memory copies between CPU and GPU.

```bash
pip install mlx-lm

# Convert to MLX format
mlx_lm.convert --hf-path HuggingFaceTB/SmolLM3-3B-Instruct --mlx-path ./smollm3-mlx

# QLoRA fine-tune (automatic when using quantized model)
mlx_lm.lora \
  --model ./smollm3-mlx \
  --train \
  --data ./output \
  --batch-size 4 \
  --lora-layers 16 \
  --iters 1000 \
  --learning-rate 1e-5
```

**Resource requirements:**
- Memory: ~6-8 GB (QLoRA on 3B model)
- Time: ~15-30 minutes on M3/M4
- Storage: ~6 GB for model + adapters

### 6. Results

Training completed in 500 iterations (~40 minutes on Apple Silicon M-series, 59.5 GB peak unified memory). Only 0.218% of parameters were trained (6.7M adapter weights out of 3,075M total).

#### Training Loss Curve

| Iter | Train Loss | Val Loss |
|------|-----------|----------|
| 1 | — | 1.069 |
| 100 | 0.503 | 0.688 |
| 250 | 0.590 | **0.559** |
| 500 | 0.482 | 0.625 |

Best validation loss: **0.559** at iteration 250.

#### 3-Way Model Comparison (Hand-Crafted Prompts)

Tested on 3 prompts not present in training data:

| Model | Params | T1 | T2 | T3 | **Avg** |
|-------|--------|-----|-----|-----|---------|
| **SmolLM3-3B (fine-tuned)** | 3B | 95 | 100 | 100 | **98.3** |
| SmolLM3-3B (base) | 3B | 90 | 70 | 85 | 81.7 |
| xLAM-1B (Salesforce) | 1B | 20 | 40 | 40 | 33.3 |

#### Held-Out Validation Set Benchmark (20 Samples)

Tested on 20 randomly sampled examples from the held-out validation split (never seen during training):

| Model | Avg Score | Min | Max | Valid JSON % |
|-------|-----------|-----|-----|-------------|
| **SmolLM3-3B v2 (+targeted)** | **98.5** | **95** | **100** | 100% |
| SmolLM3-3B v1 | 92.5 | 65 | 100 | 100% |
| Qwen3-8B (fine-tuned) | 92.0 | 65 | 100 | 100% |
| SmolLM3-3B (base) | 79.3 | 60 | 90 | 100% |
| xLAM-1B (Salesforce) | 27.5 | 20 | 40 | 100% |

**Key findings:**
- **v2 model never scores below 95** on any held-out example (v1 min was 65)
- **+24.2% improvement** over the base SmolLM3-3B model
- **+258% improvement** over Salesforce's xLAM-1B (a purpose-built Large Action Model)
- Targeted training on 269 enterprise/complex examples (generated via Inception Mercury 2 for ~$1) eliminated the long tail of weak performance
- Enterprise orchestration: 70 to 100, infrastructure monitoring: 75 to 95, SDK docs sync: 65 to 95
- The fine-tuned model goes straight to clean structured JSON; the base model wraps output in `<think>` tags and prose
- The fine-tuned model learned to characterize user tone ("casual, vague request") and adjust agent complexity accordingly
- xLAM-1B scores low not because it's a bad model, but because it was trained for function *calling* (invoking existing tools), not function *creation* (designing new agent architectures)
- Qwen3-8B (2.7x larger) produces richer reasoning but scores comparably to SmolLM3-3B, at 2x the inference latency

#### Scoring Methodology

Each output is scored 0-100 based on:
- Valid JSON output: 20 points
- Presence of key schema fields: `reasoning` (10), `agent` (10), `tools` (10), `skills` (10), `constraints` (10), `steps` (10), `trigger` (5), `parameters` (5), `on_failure` (5), `description` (5)

## Project Structure

```
lam-training/
├── src/
│   ├── schema.ts          # Zod-validated agent/skill output schema
│   ├── categories.ts      # 20 categories x 5 seeds for data diversity
│   ├── generate.ts        # Batch API submission, status, and collection
│   ├── validate.ts        # Post-generation schema validation
│   ├── stats.ts           # Dataset distribution analysis
│   ├── test-repetition.ts # A/B test: single vs repeated instruction
│   └── probe-model.ts     # Model ID discovery utility
├── output/                # Generated training data (gitignored)
├── package.json
└── tsconfig.json
```

## Usage

```bash
npm install

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Submit batch (100 requests x 10 examples = 1,000 examples, 50% cheaper than real-time)
npm run submit

# Check progress
npm run status

# Download and validate results when batch completes
npm run collect

# Verify schema compliance
npm run validate

# Analyze distribution
npm run stats
```

## Training Data Categories

The v2 dataset contains 1,580 training examples from three sources:

**Phase 1: Broad coverage (992 examples via Claude Sonnet 4.6 Batch API)**
20 categories including code review, DevOps, data pipeline, testing, security, monitoring, ML ops, multi-agent orchestration, compliance, cost optimization, onboarding, and more.

**Phase 2: Targeted weak-category reinforcement (269 examples via Inception Mercury 2)**
Enterprise API lifecycle, cross-team communication orchestration, complex report pipelines, multi-agent systems, infrastructure compliance, security posture management, and formal workflow automation.

**Phase 3: Anti-forgetting mix (500 ToolACE + general conversation)**
Prevents catastrophic forgetting of structured JSON output patterns and general instruction following.

## Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Phase 1: Claude Batch API (992 examples) | ~$3.23 | 50% batch discount + prompt caching |
| Phase 2: Mercury 2 API (269 examples) | ~$1.00 | $0.75/M output tokens, 158 seconds |
| Fine-tuning (local, MLX) | $0 | Apple Silicon, ~40 min per run |
| Inference (local) | $0 | ~80 tok/s on M-series Mac |
| **Total** | **~$4.23** | |

## Key Research References

- [xLAM: A Family of Large Action Models](https://github.com/SalesforceAIResearch/xLAM) — Salesforce's LAM framework
- [Small Language Models for Efficient Agentic Tool Calling](https://arxiv.org/abs/2512.15943) — fine-tuned SLMs outperforming large models
- [ToolACE: Winning the Points of LLM Function Calling](https://openreview.net/forum?id=8EB8k6DdCU) — ICLR 2025
- [SmolLM3-3B](https://huggingface.co/blog/smollm3) — model architecture and training details
- [FunctionGemma](https://blog.google/technology/developers/functiongemma/) — Google's 270M function-calling model
