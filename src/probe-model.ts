import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const candidates = [
  "claude-sonnet-4-6",
  "claude-sonnet-4-6-20260217",
  "claude-sonnet-4-6-20260218",
  "claude-4-6-sonnet-20260217",
  "claude-sonnet-4-5-20250514",
];

async function probe(model: string) {
  try {
    const r = await client.messages.create({
      model,
      max_tokens: 10,
      messages: [{ role: "user", content: "hi" }],
    });
    console.log(`${model} -> OK (resolved: ${r.model})`);
  } catch (e: any) {
    console.log(`${model} -> FAIL: ${e.status || "?"} ${e.error?.error?.message || e.message}`);
  }
}

async function main() {
  for (const m of candidates) {
    await probe(m);
  }
}

main();
