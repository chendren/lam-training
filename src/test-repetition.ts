import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();
const MODEL = "claude-sonnet-4-6";

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

const TEST_REQUEST =
  "Build an agent that monitors my AWS Lambda functions for cold start latency spikes and automatically adjusts provisioned concurrency";

function buildSinglePrompt(): string {
  return `${SCHEMA_INSTRUCTION}

USER REQUEST: "${TEST_REQUEST}"`;
}

function buildRepeatedPrompt(): string {
  return `${SCHEMA_INSTRUCTION}

USER REQUEST: "${TEST_REQUEST}"

Let me repeat your instructions: ${SCHEMA_INSTRUCTION}

Now generate the agent definition for this request: "${TEST_REQUEST}"`;
}

interface TestResult {
  label: string;
  toolCount: number;
  skillCount: number;
  totalSteps: number;
  constraintCount: number;
  reasoningLength: number;
  reasoningMentionsWhy: boolean;
  toolsHaveDescriptions: boolean;
  allParamsHaveTypes: boolean;
  allSkillsHaveTriggers: boolean;
  stepsHaveToolRefs: number;
  outputTokens: number;
  latencyMs: number;
  valid: boolean;
  errors: string[];
}

function stripMarkdownFences(text: string): string {
  let result = text.trim();
  if (result.startsWith("```")) {
    const firstNewline = result.indexOf("\n");
    if (firstNewline !== -1) {
      result = result.slice(firstNewline + 1);
    }
  }
  if (result.endsWith("```")) {
    result = result.slice(0, result.length - 3);
  }
  return result.trim();
}

function analyze(label: string, raw: string, outputTokens: number, latencyMs: number): TestResult {
  const errors: string[] = [];
  let parsed: any;

  try {
    parsed = JSON.parse(stripMarkdownFences(raw));
  } catch {
    return {
      label,
      toolCount: 0,
      skillCount: 0,
      totalSteps: 0,
      constraintCount: 0,
      reasoningLength: 0,
      reasoningMentionsWhy: false,
      toolsHaveDescriptions: false,
      allParamsHaveTypes: false,
      allSkillsHaveTriggers: false,
      stepsHaveToolRefs: 0,
      outputTokens,
      latencyMs,
      valid: false,
      errors: ["Failed to parse JSON"],
    };
  }

  const agent = parsed.agent;
  if (!agent) {
    errors.push("Missing agent object");
  }

  const reasoning = parsed.reasoning || "";
  const tools = agent?.tools || [];
  const skills = agent?.skills || [];
  const constraints = agent?.constraints || [];

  let totalSteps = 0;
  let stepsWithToolRefs = 0;
  let allSkillsHaveTriggers = true;

  for (const skill of skills) {
    const steps = skill.steps || [];
    totalSteps += steps.length;
    for (const step of steps) {
      if (step.tool) stepsWithToolRefs++;
    }
    if (!skill.trigger) allSkillsHaveTriggers = false;
  }

  let allParamsHaveTypes = true;
  let toolsHaveDescriptions = true;
  for (const tool of tools) {
    if (!tool.description) toolsHaveDescriptions = false;
    for (const param of tool.parameters || []) {
      if (!param.type) allParamsHaveTypes = false;
    }
  }

  // Check if reasoning explains WHY not just WHAT
  const reasoningLower = reasoning.toLowerCase();
  const reasoningMentionsWhy =
    reasoningLower.includes("because") ||
    reasoningLower.includes("since") ||
    reasoningLower.includes("allows") ||
    reasoningLower.includes("enables") ||
    reasoningLower.includes("ensures") ||
    reasoningLower.includes("in order to") ||
    reasoningLower.includes("so that") ||
    reasoningLower.includes("the reason");

  if (tools.length === 0) errors.push("No tools defined");
  if (skills.length === 0) errors.push("No skills defined");
  if (constraints.length === 0) errors.push("No constraints defined");
  if (reasoning.length < 50) errors.push("Reasoning too short");

  return {
    label,
    toolCount: tools.length,
    skillCount: skills.length,
    totalSteps,
    constraintCount: constraints.length,
    reasoningLength: reasoning.length,
    reasoningMentionsWhy,
    toolsHaveDescriptions,
    allParamsHaveTypes,
    allSkillsHaveTriggers,
    stepsHaveToolRefs: stepsWithToolRefs,
    outputTokens,
    latencyMs,
    valid: errors.length === 0,
    errors,
  };
}

async function runTest(label: string, prompt: string): Promise<TestResult> {
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

  // Debug: show first/last 200 chars if parse will fail
  const stripped = stripMarkdownFences(text);
  try {
    JSON.parse(stripped);
  } catch {
    console.log(`\n  [DEBUG ${label}] stop_reason: ${response.stop_reason}`);
    console.log(`  [DEBUG] First 300 chars: ${stripped.slice(0, 300)}`);
    console.log(`  [DEBUG] Last 200 chars: ...${stripped.slice(-200)}`);
  }

  return analyze(label, text, outputTokens, latencyMs);
}

function printResult(r: TestResult) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`  ${r.label}`);
  console.log(`${"=".repeat(60)}`);
  console.log(`  Valid:                  ${r.valid ? "YES" : "NO"}`);
  console.log(`  Tools:                  ${r.toolCount}`);
  console.log(`  Skills:                 ${r.skillCount}`);
  console.log(`  Total steps:            ${r.totalSteps}`);
  console.log(`  Steps with tool refs:   ${r.stepsHaveToolRefs}/${r.totalSteps}`);
  console.log(`  Constraints:            ${r.constraintCount}`);
  console.log(`  Reasoning length:       ${r.reasoningLength} chars`);
  console.log(`  Reasoning explains WHY: ${r.reasoningMentionsWhy ? "YES" : "NO"}`);
  console.log(`  Tools have descriptions:${r.toolsHaveDescriptions ? " YES" : " NO"}`);
  console.log(`  All params have types:  ${r.allParamsHaveTypes ? "YES" : "NO"}`);
  console.log(`  All skills have trigger:${r.allSkillsHaveTriggers ? " YES" : " NO"}`);
  console.log(`  Output tokens:          ${r.outputTokens}`);
  console.log(`  Latency:                ${r.latencyMs}ms`);
  if (r.errors.length > 0) {
    console.log(`  Errors:                 ${r.errors.join(", ")}`);
  }
}

function computeScore(r: TestResult): number {
  let score = 0;
  if (r.valid) score += 20;
  score += Math.min(r.toolCount, 5) * 4;          // max 20
  score += Math.min(r.skillCount, 4) * 5;          // max 20
  score += Math.min(r.totalSteps, 10) * 1;         // max 10
  if (r.reasoningMentionsWhy) score += 10;
  if (r.toolsHaveDescriptions) score += 5;
  if (r.allParamsHaveTypes) score += 5;
  if (r.allSkillsHaveTriggers) score += 5;
  score += Math.min(r.constraintCount, 5) * 1;     // max 5
  return score;
}

async function main() {
  console.log("\nA/B Test: Single Instruction vs. Repeated Instruction");
  console.log(`Model: ${MODEL}`);
  console.log(`Request: "${TEST_REQUEST}"\n`);

  // Run both tests
  const [single, repeated] = await Promise.all([
    runTest("SINGLE INSTRUCTION", buildSinglePrompt()),
    runTest("REPEATED INSTRUCTION", buildRepeatedPrompt()),
  ]);

  printResult(single);
  printResult(repeated);

  const singleScore = computeScore(single);
  const repeatedScore = computeScore(repeated);
  const improvement = singleScore > 0
    ? (((repeatedScore - singleScore) / singleScore) * 100).toFixed(1)
    : "N/A";

  console.log(`\n${"=".repeat(60)}`);
  console.log(`  COMPARISON`);
  console.log(`${"=".repeat(60)}`);
  console.log(`  Single score:     ${singleScore}/100`);
  console.log(`  Repeated score:   ${repeatedScore}/100`);
  console.log(`  Improvement:      ${improvement}%`);
  console.log(`  Token overhead:   +${repeated.outputTokens - single.outputTokens} output tokens`);
  console.log(`  Latency delta:    ${repeated.latencyMs - single.latencyMs}ms`);
}

main().catch(console.error);
