import { writeFileSync, readFileSync, createWriteStream } from "fs";
import { pipeline } from "stream/promises";

const TOOLACE_REPO = "Team-ACE/ToolACE";
const CONVERSATION_REPO = "yahma/alpaca-cleaned";
const SYNTHETIC_FILE = "output/training_data.jsonl";
const TRAIN_FILE = "output/train.jsonl";
const VALID_FILE = "output/valid.jsonl";

const TOOLACE_SAMPLE = 500;
const CONVERSATION_SAMPLE = 500;
const VALIDATION_SPLIT = 0.1;

function progress(label: string, current: number, total: number) {
  const pct = Math.round((current / total) * 100);
  const bar = "█".repeat(Math.floor(pct / 2)) + "░".repeat(50 - Math.floor(pct / 2));
  process.stdout.write(`\r  ${label.padEnd(25)} ${bar} ${pct}% (${current}/${total})`);
  if (current === total) process.stdout.write("\n");
}

function shuffle(arr: any[]) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Reservoir sampling: pick k items from a stream without loading all into memory
function reservoirSample<T>(items: T[], k: number): T[] {
  const result = items.slice(0, k);
  for (let i = k; i < items.length; i++) {
    const j = Math.floor(Math.random() * (i + 1));
    if (j < k) result[j] = items[i];
  }
  return result;
}

interface MLXExample {
  messages: { role: string; content: string }[];
}

// --- Load synthetic data ---
function loadSynthetic(): MLXExample[] {
  console.log("\n[1/4] Loading synthetic agent-creation data...");
  const lines = readFileSync(SYNTHETIC_FILE, "utf-8")
    .split("\n")
    .filter((l) => l.trim().length > 0);

  const examples: MLXExample[] = [];
  for (let i = 0; i < lines.length; i++) {
    examples.push(JSON.parse(lines[i]));
    if (i % 100 === 0 || i === lines.length - 1) {
      progress("Synthetic", i + 1, lines.length);
    }
  }
  return examples;
}

// --- Download JSON dataset from HuggingFace via direct file URL ---
async function downloadHFJson(label: string, url: string): Promise<any[]> {
  console.log(`  Fetching ${label}...`);
  progress(`${label} download`, 0, 1);
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
  const text = await resp.text();
  progress(`${label} download`, 1, 1);

  // Handle JSONL (one object per line) or JSON array
  const trimmed = text.trim();
  if (trimmed.startsWith("[")) {
    return JSON.parse(trimmed);
  }
  // JSONL format
  return trimmed.split("\n").filter(l => l.trim().length > 0).map(l => JSON.parse(l));
}

// --- Download and sample ToolACE ---
async function loadToolACE(): Promise<MLXExample[]> {
  console.log("\n[2/4] Downloading ToolACE from HuggingFace...");

  // Direct parquet-converted JSON via the datasets-server first-rows API (returns up to 100 rows)
  // Instead, download the raw dataset file
  const url = "https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/main/data/train.json";

  let allRows: any[];
  try {
    allRows = await downloadHFJson("ToolACE", url);
  } catch {
    // Fallback: try JSONL format
    try {
      const url2 = "https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/main/data/train.jsonl";
      allRows = await downloadHFJson("ToolACE (jsonl)", url2);
    } catch {
      // Final fallback: rows API with small pages
      console.log("  Direct download failed, using rows API...");
      allRows = [];
      for (let offset = 0; offset < 2000; offset += 100) {
        const apiUrl = `https://datasets-server.huggingface.co/rows?dataset=Team-ACE%2FToolACE&config=default&split=train&offset=${offset}&length=100`;
        try {
          const resp = await fetch(apiUrl);
          const data = await resp.json() as any;
          if (data.rows) allRows.push(...data.rows.map((r: any) => r.row));
        } catch { /* skip */ }
        progress("ToolACE API", Math.min(offset + 100, 2000), 2000);
      }
    }
  }

  console.log(`  Got ${allRows.length} rows, sampling ${TOOLACE_SAMPLE}...`);
  const sampled = reservoirSample(allRows, TOOLACE_SAMPLE);
  const examples: MLXExample[] = [];

  for (let i = 0; i < sampled.length; i++) {
    const row = sampled[i] as any;
    try {
      const conversations = row.conversations || row.messages || [];
      if (conversations.length >= 2) {
        const messages = conversations.map((msg: any) => ({
          role: msg.role === "function" ? "assistant" : msg.role,
          content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
        }));
        examples.push({ messages });
      }
    } catch { /* skip */ }
    if (i % 50 === 0 || i === sampled.length - 1) {
      progress("ToolACE convert", i + 1, sampled.length);
    }
  }

  return examples;
}

// --- Download and sample general conversation data ---
async function loadConversation(): Promise<MLXExample[]> {
  console.log("\n[3/4] Downloading Alpaca-Cleaned from HuggingFace...");

  const url = "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json";

  let allRows: any[];
  try {
    allRows = await downloadHFJson("Alpaca", url);
  } catch {
    console.log("  Direct download failed, using rows API...");
    allRows = [];
    for (let offset = 0; offset < 2000; offset += 100) {
      const apiUrl = `https://datasets-server.huggingface.co/rows?dataset=yahma%2Falpaca-cleaned&config=default&split=train&offset=${offset}&length=100`;
      try {
        const resp = await fetch(apiUrl);
        const data = await resp.json() as any;
        if (data.rows) allRows.push(...data.rows.map((r: any) => r.row));
      } catch { /* skip */ }
      progress("Alpaca API", Math.min(offset + 100, 2000), 2000);
    }
  }

  console.log(`  Got ${allRows.length} rows, sampling ${CONVERSATION_SAMPLE}...`);
  const sampled = reservoirSample(allRows, CONVERSATION_SAMPLE);
  const examples: MLXExample[] = [];

  for (let i = 0; i < sampled.length; i++) {
    const row = sampled[i] as any;
    try {
      const messages: { role: string; content: string }[] = [];
      const instruction = row.instruction || "";
      const input = row.input || "";
      const output = row.output || "";

      const userContent = input.length > 0 ? `${instruction}\n\n${input}` : instruction;
      if (userContent.length > 0) messages.push({ role: "user", content: userContent });
      if (output.length > 0) messages.push({ role: "assistant", content: output });

      if (messages.length >= 2) examples.push({ messages });
    } catch { /* skip */ }
    if (i % 50 === 0 || i === sampled.length - 1) {
      progress("Alpaca convert", i + 1, sampled.length);
    }
  }

  return examples;
}

// --- Merge, shuffle, split ---
async function main() {
  console.log("=".repeat(60));
  console.log("  LAM Training Dataset Preparation");
  console.log("=".repeat(60));

  const synthetic = loadSynthetic();
  const toolace = await loadToolACE();
  const conversation = await loadConversation();

  console.log("\n[4/4] Merging, shuffling, and splitting...");
  console.log(`  Synthetic:      ${synthetic.length}`);
  console.log(`  ToolACE:        ${toolace.length}`);
  console.log(`  Conversation:   ${conversation.length}`);

  const all = [...synthetic, ...toolace, ...conversation];
  shuffle(all);

  const splitIdx = Math.floor(all.length * (1 - VALIDATION_SPLIT));
  const train = all.slice(0, splitIdx);
  const valid = all.slice(splitIdx);

  progress("Writing train.jsonl", 0, train.length);
  const trainLines: string[] = [];
  for (let i = 0; i < train.length; i++) {
    trainLines.push(JSON.stringify(train[i]));
    if (i % 200 === 0 || i === train.length - 1) {
      progress("Writing train.jsonl", i + 1, train.length);
    }
  }
  writeFileSync(TRAIN_FILE, trainLines.join("\n") + "\n");

  progress("Writing valid.jsonl", 0, valid.length);
  const validLines: string[] = [];
  for (let i = 0; i < valid.length; i++) {
    validLines.push(JSON.stringify(valid[i]));
    if (i % 50 === 0 || i === valid.length - 1) {
      progress("Writing valid.jsonl", i + 1, valid.length);
    }
  }
  writeFileSync(VALID_FILE, validLines.join("\n") + "\n");

  console.log("\n" + "=".repeat(60));
  console.log("  Dataset Ready!");
  console.log("=".repeat(60));
  console.log(`  Train: ${train.length} examples → ${TRAIN_FILE}`);
  console.log(`  Valid: ${valid.length} examples → ${VALID_FILE}`);
  console.log(`  Total: ${all.length} examples`);
  console.log(`  Split: ${Math.round((1 - VALIDATION_SPLIT) * 100)}% train / ${Math.round(VALIDATION_SPLIT * 100)}% validation`);
}

main().catch(console.error);
