import { execFileSync } from "child_process";
import { writeFileSync, unlinkSync } from "fs";
import { LAM_SYSTEM_PROMPT } from "./schema.js";

const SMOLLM3_MODEL = "models/smollm3-3b-8bit";
const QWEN3_MODEL = "models/qwen3-8b-8bit";
const XLAM_MODEL = "models/xlam-1b-8bit";
const ADAPTER_PATH = "adapters";
const QWEN3_ADAPTER_PATH = "adapters-qwen3";

const TEST_PROMPTS = [
  "Build an agent that monitors my S3 buckets for sensitive data exposure and auto-remediates",
  "Create a simple bot that posts daily standup summaries to Slack from Jira tickets",
  "I need an agent that reviews PRs for SQL injection vulnerabilities before merge",
];

function runInference(prompt: string, model: string, adapterPath?: string): { output: string; timeMs: number } {
  const fullPrompt = `${LAM_SYSTEM_PROMPT}\n\nUser request: ${prompt}`;

  const args = [
    "--model", model,
    "--max-tokens", "2048",
    "--prompt", fullPrompt,
  ];

  if (adapterPath) {
    args.push("--adapter-path", adapterPath);
  }

  const start = Date.now();
  let result: string;
  try {
    result = execFileSync("mlx_lm.generate", args, {
      encoding: "utf-8",
      timeout: 120000,
      maxBuffer: 10 * 1024 * 1024,
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (err: any) {
    result = err.stdout || "(errored)";
  } finally {
    // cleanup
  }
  const timeMs = Date.now() - start;

  return { output: result, timeMs };
}

function hasValidJson(text: string): boolean {
  const braceStart = text.indexOf("{");
  if (braceStart === -1) return false;

  let depth = 0;
  for (let i = braceStart; i < text.length; i++) {
    if (text[i] === "{") depth++;
    if (text[i] === "}") depth--;
    if (depth === 0) {
      try {
        JSON.parse(text.slice(braceStart, i + 1));
        return true;
      } catch {
        return false;
      }
    }
  }
  return false;
}

function scoreOutput(text: string): { score: number; details: string[] } {
  const details: string[] = [];
  let score = 0;

  if (hasValidJson(text)) {
    score += 20;
    details.push("Valid JSON: YES");
  } else {
    details.push("Valid JSON: NO");
  }

  const lower = text.toLowerCase();
  const fields = [
    { name: "reasoning", weight: 10 },
    { name: "agent", weight: 10 },
    { name: "tools", weight: 10 },
    { name: "skills", weight: 10 },
    { name: "constraints", weight: 10 },
    { name: "trigger", weight: 5 },
    { name: "parameters", weight: 5 },
    { name: "steps", weight: 10 },
    { name: "on_failure", weight: 5 },
    { name: "description", weight: 5 },
  ];

  for (const field of fields) {
    if (lower.includes(`"${field.name}"`)) {
      score += field.weight;
      details.push(`${field.name}: present`);
    }
  }

  return { score, details };
}

interface ModelConfig {
  label: string;
  model: string;
  adapter?: string;
}

async function main() {
  const models: ModelConfig[] = [
    { label: "SmolLM3-3B (fine-tuned)", model: SMOLLM3_MODEL, adapter: ADAPTER_PATH },
    { label: "Qwen3-8B (fine-tuned)", model: QWEN3_MODEL, adapter: QWEN3_ADAPTER_PATH },
    { label: "xLAM-1B (Salesforce)", model: XLAM_MODEL },
  ];

  console.log("=".repeat(70));
  console.log("  LAM Inference Test: 3-Way Comparison");
  console.log("=".repeat(70));

  const allScores: number[][] = models.map(() => []);

  for (let i = 0; i < TEST_PROMPTS.length; i++) {
    const prompt = TEST_PROMPTS[i];
    console.log(`\n${"─".repeat(70)}`);
    console.log(`  Test ${i + 1}: "${prompt.slice(0, 65)}..."`);
    console.log(`${"─".repeat(70)}`);

    for (let m = 0; m < models.length; m++) {
      const cfg = models[m];
      console.log(`\n  [${cfg.label}] Generating...`);
      const result = runInference(prompt, cfg.model, cfg.adapter);
      const score = scoreOutput(result.output);
      allScores[m].push(score.score);

      console.log(`  Score: ${score.score}/100 | Time: ${(result.timeMs / 1000).toFixed(1)}s`);
      console.log(`  ${score.details.slice(0, 5).join(" | ")}`);
      console.log(`  Output preview: ${result.output.trim().slice(0, 250)}...`);
    }
  }

  // Summary
  console.log(`\n${"=".repeat(70)}`);
  console.log("  SUMMARY");
  console.log("=".repeat(70));
  console.log(`\n  ${"Model".padEnd(30)} ${TEST_PROMPTS.map((_, i) => `T${i+1}`).join("    ")}    Avg`);
  for (let m = 0; m < models.length; m++) {
    const scores = allScores[m];
    const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
    const scoreStr = scores.map(s => String(s).padStart(3)).join("    ");
    console.log(`  ${models[m].label.padEnd(30)} ${scoreStr}    ${avg.toFixed(1)}`);
  }
}

main().catch(console.error);
