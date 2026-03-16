import { execFileSync } from "child_process";
import { readFileSync } from "fs";
import { LAM_SYSTEM_PROMPT } from "./schema.js";

const SMOLLM3_MODEL = "models/smollm3-3b-8bit";
const XLAM_MODEL = "models/xlam-1b-8bit";
const ADAPTER_PATH = "adapters";
const VALID_FILE = "output/valid.jsonl";
const SAMPLE_SIZE = 20; // test 20 random validation examples

interface ModelConfig {
  label: string;
  model: string;
  adapter?: string;
}

const MODELS: ModelConfig[] = [
  { label: "SmolLM3-3B (base)", model: SMOLLM3_MODEL },
  { label: "SmolLM3-3B (fine-tuned)", model: SMOLLM3_MODEL, adapter: ADAPTER_PATH },
  { label: "xLAM-1B (Salesforce)", model: XLAM_MODEL },
];

function runInference(prompt: string, model: string, adapterPath?: string): string {
  const fullPrompt = `${LAM_SYSTEM_PROMPT}\n\nUser request: ${prompt}`;
  const args = [
    "--model", model,
    "--max-tokens", "2048",
    "--prompt", fullPrompt,
  ];
  if (adapterPath) args.push("--adapter-path", adapterPath);

  try {
    return execFileSync("mlx_lm.generate", args, {
      encoding: "utf-8",
      timeout: 120000,
      maxBuffer: 10 * 1024 * 1024,
      stdio: ["pipe", "pipe", "pipe"],
    });
  } catch (err: any) {
    return err.stdout || "(errored)";
  }
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

function scoreOutput(text: string): number {
  let score = 0;
  if (hasValidJson(text)) score += 20;

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
    if (lower.includes(`"${field.name}"`)) score += field.weight;
  }
  return score;
}

function extractUserRequests(): string[] {
  const lines = readFileSync(VALID_FILE, "utf-8").split("\n").filter(l => l.trim());
  const requests: string[] = [];

  for (const line of lines) {
    const ex = JSON.parse(line);
    // Only use synthetic examples (they have our system prompt), skip ToolACE/Alpaca
    if (ex.messages && ex.messages.length === 3 && ex.messages[0].content.includes("Large Action Model")) {
      requests.push(ex.messages[1].content);
    }
  }
  return requests;
}

function shuffle<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

async function main() {
  const allRequests = extractUserRequests();
  const requests = shuffle(allRequests).slice(0, SAMPLE_SIZE);

  console.log("=".repeat(70));
  console.log("  Validation Set Test: Held-Out Examples (not used in training)");
  console.log("=".repeat(70));
  console.log(`  Validation examples available: ${allRequests.length}`);
  console.log(`  Testing: ${requests.length} random samples`);

  const scores: Map<string, number[]> = new Map();
  for (const m of MODELS) scores.set(m.label, []);

  for (let i = 0; i < requests.length; i++) {
    const request = requests[i];
    const shortReq = request.length > 70 ? request.slice(0, 70) + "..." : request;
    console.log(`\n  [${i + 1}/${requests.length}] "${shortReq}"`);

    const rowScores: string[] = [];
    for (const cfg of MODELS) {
      const output = runInference(request, cfg.model, cfg.adapter);
      const s = scoreOutput(output);
      scores.get(cfg.label)!.push(s);
      rowScores.push(`${cfg.label.split("(")[1]?.replace(")", "") || cfg.label}: ${s}`);
    }
    console.log(`    ${rowScores.join(" | ")}`);
  }

  // Summary
  const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

  console.log(`\n${"=".repeat(70)}`);
  console.log("  RESULTS ON HELD-OUT VALIDATION SET");
  console.log("=".repeat(70));
  console.log(`\n  ${"Model".padEnd(30)} ${"Avg".padEnd(8)} ${"Min".padEnd(8)} ${"Max".padEnd(8)} Valid JSON %`);

  for (const cfg of MODELS) {
    const s = scores.get(cfg.label)!;
    const a = avg(s);
    const min = Math.min(...s);
    const max = Math.max(...s);
    const validPct = Math.round(s.filter(x => x >= 20).length / s.length * 100);
    console.log(`  ${cfg.label.padEnd(30)} ${a.toFixed(1).padEnd(8)} ${String(min).padEnd(8)} ${String(max).padEnd(8)} ${validPct}%`);
  }
}

main().catch(console.error);
