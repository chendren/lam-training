import Anthropic from "@anthropic-ai/sdk";
import { writeFileSync, readFileSync, existsSync } from "fs";
import { CATEGORIES, type Category } from "./categories.js";
import {
  TrainingExampleSchema,
  LAM_SYSTEM_PROMPT,
  type TrainingExample,
  type MLXTrainingExample,
} from "./schema.js";

const client = new Anthropic();

const MODEL = "claude-sonnet-4-6";
const BATCH_SIZE = 4; // examples per request — each example is ~4K output tokens, 4 fits in 16K
const OUTPUT_FILE = "output/training_data.jsonl";
const BATCH_ID_FILE = "output/batch_id.txt";
const RAW_RESULTS_FILE = "output/batch_results_raw.jsonl";

// --- Cached system prompt (identical across all requests in the batch) ---

const SYSTEM_PROMPT = `You are an expert at creating AI agent definitions with tools, skills, and constraints. You generate high-quality training examples for a Large Action Model that creates AI agents from natural language requests.

RULES:
- tool names use snake_case
- skill names use kebab-case
- steps should be concrete actions, not vague descriptions
- constraints should be specific and enforceable
- reasoning should explain WHY this architecture, not just WHAT it does
- each tool must have at least 1 parameter
- each skill must have at least 2 steps
- vary the number of tools (1-5) and skills (1-4) based on complexity

Each example must be a JSON object matching this structure:
{
  "user_request": "the natural language request",
  "reasoning": "why this agent architecture fits the request",
  "agent": {
    "name": "agent-name",
    "description": "what the agent does",
    "role": "the agent's primary role in one phrase",
    "tools": [
      {
        "name": "tool_name",
        "description": "what the tool does",
        "parameters": [
          { "name": "param", "type": "string|number|boolean|array|object", "description": "...", "required": true }
        ],
        "returns": "description of return value"
      }
    ],
    "skills": [
      {
        "name": "skill-name",
        "description": "what the skill accomplishes",
        "trigger": "when this skill activates",
        "inputs": [
          { "name": "input", "type": "string", "description": "...", "required": true }
        ],
        "steps": [
          { "action": "what happens", "tool": "tool_name", "input": { "param": "value_source" } },
          { "action": "next step", "on_failure": "stop" }
        ],
        "output": "what the skill produces"
      }
    ],
    "constraints": ["specific behavioral constraint"]
  }
}

Return ONLY valid JSON. No markdown fences, no commentary.`;

function buildUserPrompt(category: Category, seed: string, batchSize: number): string {
  const instruction = `Generate ${batchSize} diverse training examples for a Large Action Model.

CATEGORY: ${category.name}
CATEGORY DESCRIPTION: ${category.description}
SEED REQUEST: "${seed}"

For each example, create a DIFFERENT user request that is related to the seed but varies in:
- Specificity (some vague, some very detailed)
- Complexity (some single-skill, some multi-skill)
- Industry context (healthcare, fintech, e-commerce, SaaS, etc.)
- Technical stack mentioned (AWS, GCP, Node.js, Python, etc.)
- Tone (casual "build me a thing" vs formal "implement an automated system")

For each user request, generate a complete agent definition as a JSON object with reasoning, agent name, description, role, tools, skills, and constraints.

Respond with a JSON array of exactly ${batchSize} objects.`;

  // Instruction repetition technique
  return `${instruction}

Let me repeat your instructions: ${instruction}

Now generate the ${batchSize} training examples for category "${category.name}" with seed "${seed}".`;
}

function stripMarkdownFences(text: string): string {
  let result = text.trim();
  if (result.startsWith("```")) {
    const firstNewline = result.indexOf("\n");
    if (firstNewline !== -1) result = result.slice(firstNewline + 1);
  }
  if (result.endsWith("```")) result = result.slice(0, result.length - 3);
  return result.trim();
}

function toMLXFormat(example: TrainingExample): MLXTrainingExample {
  return {
    messages: [
      { role: "system", content: LAM_SYSTEM_PROMPT },
      { role: "user", content: example.user_request },
      {
        role: "assistant",
        content: JSON.stringify(
          { reasoning: example.reasoning, agent: example.agent },
          null,
          2
        ),
      },
    ],
  };
}

// --- Step 1: Submit the batch ---

async function submitBatch() {
  console.log("\nBuilding batch requests...");

  // 20 categories x 5 seeds x ~3 rounds = 250 requests x 4 examples = 1000 total
  const requests: any[] = [];
  const TARGET_REQUESTS = 250;

  let round = 0;
  while (requests.length < TARGET_REQUESTS) {
    for (const category of CATEGORIES) {
      for (let seedIdx = 0; seedIdx < category.seeds.length; seedIdx++) {
        if (requests.length >= TARGET_REQUESTS) break;
        const seed = category.seeds[seedIdx];
        const customId = `${category.name}_seed${seedIdx}_r${round}`;

      requests.push({
        custom_id: customId,
        params: {
          model: MODEL,
          max_tokens: 16384,
          temperature: 0.9,
          system: [
            {
              type: "text" as const,
              text: SYSTEM_PROMPT,
              cache_control: { type: "ephemeral" as const },
            },
          ],
          messages: [
            {
              role: "user" as const,
              content: buildUserPrompt(category, seed, BATCH_SIZE),
            },
          ],
        },
      });
      }
    }
    round++;
  }

  console.log(`Submitting batch: ${requests.length} requests x ${BATCH_SIZE} examples each = ${requests.length * BATCH_SIZE} target examples`);
  console.log(`Model: ${MODEL}`);
  console.log(`Cost: 50% of standard API pricing\n`);

  const batch = await client.messages.batches.create({ requests });

  console.log(`Batch created!`);
  console.log(`  ID: ${batch.id}`);
  console.log(`  Status: ${batch.processing_status}`);

  writeFileSync(BATCH_ID_FILE, batch.id);
  console.log(`\nBatch ID saved to ${BATCH_ID_FILE}`);
  console.log(`Run 'npm run status' to check progress.`);
  console.log(`Run 'npm run collect' to download results when done.`);
}

// --- Step 2: Check status ---

async function checkStatus() {
  const batchId = readFileSync(BATCH_ID_FILE, "utf-8").trim();
  const batch = await client.messages.batches.retrieve(batchId);

  console.log(`\nBatch: ${batch.id}`);
  console.log(`  Status: ${batch.processing_status}`);
  console.log(`  Requests: ${JSON.stringify(batch.request_counts)}`);

  if (batch.processing_status === "ended") {
    console.log(`\nBatch complete! Run 'npm run collect' to download results.`);
  } else {
    console.log(`\nStill processing. Check again in a few minutes.`);
  }
}

// --- Step 3: Collect results ---

async function collectResults() {
  const batchId = readFileSync(BATCH_ID_FILE, "utf-8").trim();

  console.log(`\nDownloading results for batch: ${batchId}`);

  const batch = await client.messages.batches.retrieve(batchId);
  if (batch.processing_status !== "ended") {
    console.log(`Batch not done yet. Status: ${batch.processing_status}`);
    console.log(`Counts: ${JSON.stringify(batch.request_counts)}`);
    return;
  }

  let totalExamples = 0;
  let totalFailed = 0;
  let totalValidationErrors = 0;

  const results = await client.messages.batches.results(batchId);

  for await (const entry of results) {
    // Save raw result
    writeFileSync(RAW_RESULTS_FILE, JSON.stringify(entry) + "\n", { flag: "a" });

    if (entry.result.type !== "succeeded") {
      console.warn(`  [${entry.custom_id}] FAILED: ${entry.result.type}`);
      totalFailed++;
      continue;
    }

    const message = entry.result.message;
    const text = message.content[0].type === "text" ? message.content[0].text : "";

    try {
      const cleaned = stripMarkdownFences(text);
      const parsed = JSON.parse(cleaned);

      if (!Array.isArray(parsed)) {
        console.warn(`  [${entry.custom_id}] Not an array`);
        totalFailed++;
        continue;
      }

      const validExamples: string[] = [];
      for (const item of parsed) {
        const result = TrainingExampleSchema.safeParse(item);
        if (result.success) {
          validExamples.push(JSON.stringify(toMLXFormat(result.data)));
        } else {
          totalValidationErrors++;
        }
      }

      if (validExamples.length > 0) {
        writeFileSync(OUTPUT_FILE, validExamples.join("\n") + "\n", { flag: "a" });
      }

      totalExamples += validExamples.length;
      console.log(`  [${entry.custom_id}] ${validExamples.length}/${parsed.length} valid`);
    } catch (err) {
      console.warn(`  [${entry.custom_id}] Parse error: ${err instanceof Error ? err.message : err}`);
      totalFailed++;
    }
  }

  console.log(`\nResults collected!`);
  console.log(`  Total valid examples: ${totalExamples}`);
  console.log(`  Failed requests: ${totalFailed}`);
  console.log(`  Validation errors: ${totalValidationErrors}`);
  console.log(`  Output: ${OUTPUT_FILE}`);
}

// --- CLI ---

const command = process.argv[2] || "submit";

switch (command) {
  case "submit":
    submitBatch().catch(console.error);
    break;
  case "status":
    checkStatus().catch(console.error);
    break;
  case "collect":
    collectResults().catch(console.error);
    break;
  default:
    console.log("Usage: tsx src/generate.ts [submit|status|collect]");
}
