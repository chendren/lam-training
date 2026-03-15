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

We use Claude to generate high-quality training pairs with two key techniques:

**Instruction Repetition:** Research shows repeating the instruction in the prompt raises LLM adherence from ~21% to ~97% for structured tasks. Our A/B test confirmed a 7.1% quality improvement with repeated instructions (85 → 91/100), and the repeated variant was the only one that produced reasoning explaining *why* the architecture fits.

**Prompt Caching:** The system prompt (schema definition + rules) is identical across all 100 batch requests. By using `cache_control: { type: "ephemeral" }`, 99 of 100 requests read from cache at 90% discount.

**Batch API:** 50% cost reduction vs real-time API, no rate limits, Anthropic handles queuing and retries.

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

### 6. Evaluate

| Benchmark | What It Measures | Target |
|-----------|-----------------|--------|
| ToolBench pass rate | End-to-end tool calling | >75% |
| BFCL | Function calling accuracy | >80% |
| Schema adherence | Valid JSON output rate | >95% |
| Custom eval | Agent definition quality | Human review |

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

The 1,000 examples span 20 categories to ensure the model generalizes across domains:

| Category | Examples |
|----------|----------|
| Code Review | PR security scanning, style enforcement, anti-pattern detection |
| DevOps Automation | Blue-green deploys, canary releases, auto-scaling |
| Data Pipeline | ETL orchestration, schema validation, deduplication |
| Testing | Unit test generation, regression detection, load testing |
| Security | Dependency scanning, secrets detection, IAM auditing |
| Monitoring & Alerting | Latency tracking, anomaly detection, incident routing |
| ML Ops | Model drift detection, A/B testing, hyperparameter optimization |
| Multi-Agent Orchestration | Planner-worker systems, debate agents, pipeline chains |
| ... and 12 more | Documentation, compliance, cost optimization, onboarding, etc. |

## Cost Breakdown

| Item | Cost |
|------|------|
| Training data generation (Batch API) | ~$4-5 |
| Fine-tuning (local, MLX) | $0 |
| Inference (local) | $0 |
| **Total** | **~$5** |

## Key Research References

- [xLAM: A Family of Large Action Models](https://github.com/SalesforceAIResearch/xLAM) — Salesforce's LAM framework
- [Small Language Models for Efficient Agentic Tool Calling](https://arxiv.org/abs/2512.15943) — fine-tuned SLMs outperforming large models
- [ToolACE: Winning the Points of LLM Function Calling](https://openreview.net/forum?id=8EB8k6DdCU) — ICLR 2025
- [SmolLM3-3B](https://huggingface.co/blog/smollm3) — model architecture and training details
- [FunctionGemma](https://blog.google/technology/developers/functiongemma/) — Google's 270M function-calling model
