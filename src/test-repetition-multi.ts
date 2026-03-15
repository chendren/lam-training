import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();
const MODEL = "claude-sonnet-4-6";
const RUNS = 3;

const SCHEMA_INSTRUCTION = `You are an expert at creating AI agent definitions with tools, skills, and constraints.

Generate a training example for a Large Action Model. Given a user request, produce a complete agent definition.

RULES:
- tool names use snake_case
- skill names use kebab-case
- steps should be concrete actions, not vague descriptions
- constraints should be specific and enforceable
- reasoning should explain WHY this architecture, not just WHAT it does
- each tool must have at least 1 parameter
- each skill must have at least 2 steps
- vary the number of tools (1-5) and skills (1-4) based on complexity

Respond with a JSON object matching this structure:
{
  "user_request": "the original request",
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

Return ONLY the JSON object. No markdown fences, no commentary.`;

const REQUESTS = [
  "Build an agent that monitors my AWS Lambda functions for cold start latency spikes and automatically adjusts provisioned concurrency",
  "Create a code review bot that checks PRs for SQL injection vulnerabilities",
  "I need an agent that aggregates customer feedback from Zendesk, Intercom, and email into weekly theme reports",
];

function stripMarkdownFences(text: string): string {
  let result = text.trim();
  if (result.startsWith("```")) {
    const firstNewline = result.indexOf("\n");
    if (firstNewline !== -1) result = result.slice(firstNewline + 1);
  }
  if (result.endsWith("```")) result = result.slice(0, result.length - 3);
  return result.trim();
}

function buildSinglePrompt(request: string): string {
  return `${SCHEMA_INSTRUCTION}\n\nUSER REQUEST: "${request}"`;
}

function buildRepeatedPrompt(request: string): string {
  return `${SCHEMA_INSTRUCTION}\n\nUSER REQUEST: "${request}"\n\nLet me repeat your instructions:\n\n${SCHEMA_INSTRUCTION}\n\nNow generate the agent definition for this request: "${request}"`;
}

interface Metrics {
  valid: boolean;
  toolCount: number;
  skillCount: number;
  totalSteps: number;
  stepsWithToolRefs: number;
  constraintCount: number;
  reasoningLength: number;
  reasoningExplainsWhy: boolean;
  allParamsTyped: boolean;
  allSkillsTriggered: boolean;
  outputTokens: number;
  latencyMs: number;
}

function score(m: Metrics): number {
  let s = 0;
  if (m.valid) s += 20;
  s += Math.min(m.toolCount, 5) * 4;
  s += Math.min(m.skillCount, 4) * 5;
  s += Math.min(m.totalSteps, 10);
  if (m.reasoningExplainsWhy) s += 10;
  if (m.allParamsTyped) s += 5;
  if (m.allSkillsTriggered) s += 5;
  s += Math.min(m.constraintCount, 5);
  return s;
}

async function measure(prompt: string): Promise<Metrics> {
  const start = Date.now();
  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 8192,
    temperature: 0.7,
    messages: [{ role: "user", content: prompt }],
  });
  const latencyMs = Date.now() - start;
  const text = response.content[0].type === "text" ? response.content[0].text : "";
  const outputTokens = response.usage.output_tokens;

  try {
    const parsed = JSON.parse(stripMarkdownFences(text));
    const agent = parsed.agent || {};
    const reasoning = parsed.reasoning || "";
    const tools = agent.tools || [];
    const skills = agent.skills || [];

    let totalSteps = 0;
    let stepsWithToolRefs = 0;
    let allSkillsTriggered = true;
    for (const skill of skills) {
      const steps = skill.steps || [];
      totalSteps += steps.length;
      for (const step of steps) { if (step.tool) stepsWithToolRefs++; }
      if (!skill.trigger) allSkillsTriggered = false;
    }

    let allParamsTyped = true;
    for (const tool of tools) {
      for (const p of tool.parameters || []) { if (!p.type) allParamsTyped = false; }
    }

    const rl = reasoning.toLowerCase();
    const reasoningExplainsWhy =
      rl.includes("because") || rl.includes("since") || rl.includes("allows") ||
      rl.includes("enables") || rl.includes("ensures") || rl.includes("in order to") ||
      rl.includes("so that");

    return {
      valid: true, toolCount: tools.length, skillCount: skills.length,
      totalSteps, stepsWithToolRefs, constraintCount: (agent.constraints || []).length,
      reasoningLength: reasoning.length, reasoningExplainsWhy, allParamsTyped,
      allSkillsTriggered, outputTokens, latencyMs,
    };
  } catch {
    return {
      valid: false, toolCount: 0, skillCount: 0, totalSteps: 0, stepsWithToolRefs: 0,
      constraintCount: 0, reasoningLength: 0, reasoningExplainsWhy: false,
      allParamsTyped: false, allSkillsTriggered: false, outputTokens, latencyMs,
    };
  }
}

async function main() {
  console.log(`\nA/B Test: Single vs Repeated Instruction (${RUNS} runs x ${REQUESTS.length} prompts = ${RUNS * REQUESTS.length} samples per variant)`);
  console.log(`Model: ${MODEL}\n`);

  const singleScores: number[] = [];
  const repeatedScores: number[] = [];
  const singleMetrics: Metrics[] = [];
  const repeatedMetrics: Metrics[] = [];

  for (let run = 0; run < RUNS; run++) {
    for (const request of REQUESTS) {
      console.log(`Run ${run + 1}/${RUNS}: "${request.slice(0, 60)}..."`);

      // Run sequentially to avoid rate limits muddying latency numbers
      const sResult = await measure(buildSinglePrompt(request));
      const rResult = await measure(buildRepeatedPrompt(request));

      const sScore = score(sResult);
      const rScore = score(rResult);

      console.log(`  Single: ${sScore}/100 (${sResult.valid ? "valid" : "INVALID"})  Repeated: ${rScore}/100 (${rResult.valid ? "valid" : "INVALID"})`);

      singleScores.push(sScore);
      repeatedScores.push(rScore);
      singleMetrics.push(sResult);
      repeatedMetrics.push(rResult);
    }
  }

  const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const avgS = avg(singleScores);
  const avgR = avg(repeatedScores);
  const improvement = ((avgR - avgS) / avgS * 100).toFixed(1);

  const avgTokensS = avg(singleMetrics.map(m => m.outputTokens));
  const avgTokensR = avg(repeatedMetrics.map(m => m.outputTokens));
  const avgLatencyS = avg(singleMetrics.map(m => m.latencyMs));
  const avgLatencyR = avg(repeatedMetrics.map(m => m.latencyMs));
  const validS = singleMetrics.filter(m => m.valid).length;
  const validR = repeatedMetrics.filter(m => m.valid).length;
  const whyS = singleMetrics.filter(m => m.reasoningExplainsWhy).length;
  const whyR = repeatedMetrics.filter(m => m.reasoningExplainsWhy).length;
  const n = singleScores.length;

  console.log(`\n${"=".repeat(60)}`);
  console.log(`  AGGREGATE RESULTS (${n} samples per variant)`);
  console.log(`${"=".repeat(60)}`);
  console.log(`                    SINGLE      REPEATED    DELTA`);
  console.log(`  Avg score:        ${avgS.toFixed(1)}         ${avgR.toFixed(1)}         ${improvement}%`);
  console.log(`  Valid parses:     ${validS}/${n}          ${validR}/${n}`);
  console.log(`  Reasoning WHY:    ${whyS}/${n}          ${whyR}/${n}`);
  console.log(`  Avg tools:        ${avg(singleMetrics.map(m=>m.toolCount)).toFixed(1)}          ${avg(repeatedMetrics.map(m=>m.toolCount)).toFixed(1)}`);
  console.log(`  Avg skills:       ${avg(singleMetrics.map(m=>m.skillCount)).toFixed(1)}          ${avg(repeatedMetrics.map(m=>m.skillCount)).toFixed(1)}`);
  console.log(`  Avg steps:        ${avg(singleMetrics.map(m=>m.totalSteps)).toFixed(1)}         ${avg(repeatedMetrics.map(m=>m.totalSteps)).toFixed(1)}`);
  console.log(`  Avg constraints:  ${avg(singleMetrics.map(m=>m.constraintCount)).toFixed(1)}          ${avg(repeatedMetrics.map(m=>m.constraintCount)).toFixed(1)}`);
  console.log(`  Avg output tokens:${avgTokensS.toFixed(0)}        ${avgTokensR.toFixed(0)}`);
  console.log(`  Avg latency:      ${(avgLatencyS/1000).toFixed(1)}s        ${(avgLatencyR/1000).toFixed(1)}s`);

  // Per-sample breakdown
  console.log(`\n  Per-sample scores:`);
  console.log(`  ${"Sample".padEnd(8)} ${"Single".padEnd(8)} ${"Repeated".padEnd(10)} Winner`);
  for (let i = 0; i < n; i++) {
    const winner = singleScores[i] > repeatedScores[i] ? "SINGLE" :
                   repeatedScores[i] > singleScores[i] ? "REPEATED" : "TIE";
    console.log(`  ${String(i+1).padEnd(8)} ${String(singleScores[i]).padEnd(8)} ${String(repeatedScores[i]).padEnd(10)} ${winner}`);
  }

  const repeatedWins = singleScores.filter((s, i) => repeatedScores[i] > s).length;
  const singleWins = singleScores.filter((s, i) => repeatedScores[i] < s).length;
  const ties = n - repeatedWins - singleWins;
  console.log(`\n  Wins: Single ${singleWins} | Repeated ${repeatedWins} | Ties ${ties}`);
}

main().catch(console.error);
