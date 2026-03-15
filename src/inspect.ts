import { readFileSync } from "fs";
import { TrainingExampleSchema } from "./schema.js";

function stripMarkdownFences(text: string): string {
  let result = text.trim();
  if (result.startsWith("```")) {
    const firstNewline = result.indexOf("\n");
    if (firstNewline !== -1) result = result.slice(firstNewline + 1);
  }
  if (result.endsWith("```")) result = result.slice(0, result.length - 3);
  return result.trim();
}

const lines = readFileSync("output/batch_results_raw.jsonl", "utf-8")
  .split("\n")
  .filter((l) => l.trim().length > 0);

// Collect all validation error paths
const errorCounts = new Map<string, number>();
let parseFails = 0;
let totalItems = 0;

for (const line of lines) {
  const entry = JSON.parse(line);
  if (entry.result.type !== "succeeded") continue;

  const text = entry.result.message.content[0].text;
  let cleaned: string;
  try {
    cleaned = stripMarkdownFences(text);
    const parsed = JSON.parse(cleaned);
    if (!Array.isArray(parsed)) continue;

    for (const item of parsed) {
      totalItems++;
      const result = TrainingExampleSchema.safeParse(item);
      if (!result.success) {
        for (const issue of result.error.issues) {
          const key = `${issue.path.join(".")} | ${issue.code} | ${issue.message}`;
          errorCounts.set(key, (errorCounts.get(key) || 0) + 1);
        }
      }
    }
  } catch {
    parseFails++;
  }
}

console.log(`Total items examined: ${totalItems}`);
console.log(`JSON parse failures: ${parseFails}\n`);

const sorted = [...errorCounts.entries()].sort((a, b) => b[1] - a[1]);
console.log("Top 20 validation errors:");
for (const [key, count] of sorted.slice(0, 20)) {
  console.log(`  ${count}x  ${key}`);
}

// Show a raw example that fails
console.log("\n=== SAMPLE FAILING EXAMPLE ===");
for (const line of lines.slice(0, 5)) {
  const entry = JSON.parse(line);
  if (entry.result.type !== "succeeded") continue;
  const text = entry.result.message.content[0].text;
  try {
    const cleaned = stripMarkdownFences(text);
    const parsed = JSON.parse(cleaned);
    if (!Array.isArray(parsed)) continue;
    for (const item of parsed) {
      const result = TrainingExampleSchema.safeParse(item);
      if (!result.success) {
        console.log(JSON.stringify(item, null, 2).slice(0, 3000));
        console.log("\nErrors:");
        for (const issue of result.error.issues) {
          console.log(`  ${issue.path.join(".")} -> ${issue.message}`);
        }
        process.exit(0);
      }
    }
  } catch { /* skip */ }
}
