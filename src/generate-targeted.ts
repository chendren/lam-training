import OpenAI from "openai";
import { writeFileSync, existsSync, readFileSync } from "fs";
import {
  TrainingExampleSchema,
  LAM_SYSTEM_PROMPT,
  type TrainingExample,
  type MLXTrainingExample,
} from "./schema.js";

const client = new OpenAI({
  apiKey: "sk_25b4d76b486c79d5e6b706e2b9c83b24",
  baseURL: "https://api.inceptionlabs.ai/v1",
});

const MODEL = "mercury-2";
const MAX_TOKENS = 10000;
const TEMPERATURE = 0.9;
const MAX_CONCURRENT = 10;
const OUTPUT_FILE = "output/targeted_training_data.jsonl";
const PROGRESS_FILE = "output/targeted_progress.json";

// --- Weak category seeds: enterprise-grade, multi-system, formal complex requests ---

const TARGETED_SEEDS = [
  // Enterprise API lifecycle
  "Implement an automated API lifecycle management system for our SaaS platform that handles versioning, deprecation notices, consumer migration, and backward compatibility validation across 15 microservices",
  "Build an enterprise API governance agent that enforces OpenAPI spec compliance, tracks breaking changes across teams, and auto-generates migration guides when endpoints change",
  "Create a system that monitors API contract adherence across our microservices mesh, detects schema drift, and coordinates rollback if consumer tests fail after deployment",
  "We need an agent that manages the full lifecycle of our REST APIs from design review through deprecation, including automated consumer notification and traffic migration",

  // Enterprise cross-team communication orchestration
  "Implement an enterprise-grade cross-team communication orchestration system that routes incident updates across Slack, PagerDuty, Jira, and email based on severity, team ownership, and escalation policies",
  "Build a formal communication agent for our 200-person engineering org that coordinates release announcements, manages cross-team dependencies, and tracks acknowledgment across Slack, email, and Confluence",
  "Create a multi-channel notification orchestrator for our platform team that manages alert fatigue by deduplicating, correlating, and routing messages across 8 communication channels based on on-call schedules",
  "We need a system that coordinates handoffs between our US, EU, and APAC engineering teams during incident response, maintaining context across time zones and communication platforms",

  // Complex report pipelines
  "Implement a fully automated report pipeline for our SaaS product that aggregates data from Stripe, Mixpanel, PostgreSQL, and Salesforce into weekly executive dashboards with anomaly highlighting",
  "Build a compliance reporting agent for our fintech platform that generates SOC2 evidence packages by collecting audit logs, access reviews, and change management records from 6 different systems",
  "Create an automated quarterly business review generator that pulls metrics from our data warehouse, correlates them with OKR progress in Notion, and produces executive-ready slide decks",
  "We need an agent that produces daily operational reports by joining data from CloudWatch, Datadog, PagerDuty, and Jira, highlighting SLA breaches and trending issues",

  // Multi-agent orchestration systems
  "Design a multi-agent system where a planner agent decomposes customer support tickets into subtasks, assigns them to specialist agents for billing, technical, and account issues, and a supervisor agent validates resolution quality",
  "Build an orchestrator that coordinates between a code analysis agent, a documentation agent, and a test generation agent to perform comprehensive PR reviews with consolidated feedback",
  "Create a pipeline of agents for our data platform: an ingestion agent validates and loads data, a quality agent runs checks, a transformation agent applies business rules, and an alerting agent notifies stakeholders of anomalies",
  "Implement a debate-style architecture where two competing agents propose different architectural solutions to a technical problem, a judge agent evaluates trade-offs, and a synthesizer produces a final recommendation",

  // Complex infrastructure and compliance
  "Build an automated infrastructure compliance agent for our healthcare platform that continuously validates HIPAA controls across AWS accounts, generates gap analysis reports, and creates remediation tickets with priority scoring",
  "Create a cost optimization orchestrator for our multi-cloud deployment that analyzes spend across AWS, GCP, and Azure, identifies reserved instance opportunities, rightsizes compute, and projects savings with confidence intervals",
  "Implement an automated change management system for our regulated fintech environment that validates change requests against compliance policies, coordinates approval workflows, and maintains audit trails across ServiceNow and Jira",
  "We need an agent that manages database schema migrations across our 12 PostgreSQL instances, validates backward compatibility, coordinates blue-green deployment of schema changes, and auto-rolls back if integration tests fail",

  // Formal onboarding and knowledge management
  "Implement an automated developer onboarding system for our enterprise platform that provisions accounts across 8 tools, generates personalized learning paths from our internal wiki, assigns mentor-mentee pairings based on expertise gaps, and tracks 30-60-90 day milestones",
  "Build a knowledge management agent for our 500-person engineering org that indexes Confluence, GitHub, Slack, and recorded meetings, detects knowledge silos, and proactively surfaces relevant documentation when engineers start working on unfamiliar codebases",
  "Create an automated skills assessment agent that evaluates new hires against our engineering competency matrix, identifies training needs, and generates individualized development plans integrated with our LMS",
  "We need a formal system that manages our internal API documentation lifecycle, detects when docs drift from implementation, assigns documentation tasks to the right team, and validates examples actually compile",

  // Security and compliance orchestration
  "Build a security posture management agent for our multi-account AWS organization that runs CIS benchmarks, correlates findings with our risk register, prioritizes remediation by blast radius, and tracks fix verification",
  "Implement an automated vendor security assessment agent that processes SOC2 reports, questionnaire responses, and penetration test results, scores vendor risk, and triggers enhanced monitoring for high-risk integrations",
  "Create a secrets rotation orchestrator that manages credential lifecycle across our infrastructure, coordinates rotation with dependent services, validates connectivity after rotation, and alerts if any service fails health checks",
  "We need an agent that monitors our SaaS platform for GDPR data subject requests, coordinates data deletion across 9 microservices, generates compliance certificates, and maintains audit logs for regulatory review",

  // Complex workflow automation
  "Build an automated invoice processing pipeline for our procurement department that extracts data from PDF invoices, validates against purchase orders, routes for multi-level approval based on amount thresholds, and posts to our ERP system",
  "Create a contract lifecycle management agent that monitors contract expiration dates, triggers renewal workflows 90 days before expiry, coordinates legal review, and escalates if terms change significantly from the original agreement",
];

const SYSTEM_INSTRUCTION = `You are an expert at creating AI agent definitions with tools, skills, and constraints. Generate a training example for a Large Action Model.

Given a user request, produce a complete agent definition as a JSON object.

RULES:
- tool names use snake_case
- skill names use kebab-case
- steps should be concrete actions, not vague descriptions
- constraints should be specific and enforceable
- reasoning should explain WHY this architecture fits, not just WHAT it does
- each tool must have at least 1 parameter
- each skill must have at least 2 steps
- include 2-5 tools and 1-4 skills based on complexity
- include 3-8 constraints

Return a JSON object with this structure:
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
}`;

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

function stripMarkdownFences(text: string): string {
  let result = text.trim();
  if (result.startsWith("```")) {
    const firstNewline = result.indexOf("\n");
    if (firstNewline !== -1) result = result.slice(firstNewline + 1);
  }
  if (result.endsWith("```")) result = result.slice(0, result.length - 3);
  return result.trim();
}

interface Progress {
  completed: number;
  completedSeeds: string[];
}

function loadProgress(): Progress {
  if (existsSync(PROGRESS_FILE)) {
    return JSON.parse(readFileSync(PROGRESS_FILE, "utf-8"));
  }
  return { completed: 0, completedSeeds: [] };
}

function saveProgress(progress: Progress) {
  writeFileSync(PROGRESS_FILE, JSON.stringify(progress, null, 2));
}

async function generateOne(seed: string): Promise<TrainingExample | null> {
  // Instruction repetition technique
  const userPrompt = `${SYSTEM_INSTRUCTION}

USER REQUEST: "${seed}"

Let me repeat your instructions: ${SYSTEM_INSTRUCTION}

Now generate the agent definition for this request: "${seed}"`;

  try {
    const response = await client.chat.completions.create({
      model: MODEL,
      max_tokens: MAX_TOKENS,
      temperature: TEMPERATURE,
      messages: [{ role: "user", content: userPrompt }],
      response_format: { type: "json_object" },
    });

    const text = response.choices[0]?.message?.content || "";
    const cleaned = stripMarkdownFences(text);
    const parsed = JSON.parse(cleaned);

    const result = TrainingExampleSchema.safeParse(parsed);
    if (result.success) {
      return result.data;
    } else {
      console.warn(`  Validation failed: ${result.error.issues[0]?.path.join(".")} - ${result.error.issues[0]?.message}`);
      return null;
    }
  } catch (err) {
    console.error(`  Error: ${err instanceof Error ? err.message : err}`);
    return null;
  }
}

async function runWithConcurrency<T>(
  tasks: (() => Promise<T>)[],
  maxConcurrent: number
): Promise<T[]> {
  const results: T[] = [];
  let index = 0;

  async function runNext(): Promise<void> {
    while (index < tasks.length) {
      const currentIndex = index++;
      results[currentIndex] = await tasks[currentIndex]();
    }
  }

  const workers = Array.from(
    { length: Math.min(maxConcurrent, tasks.length) },
    () => runNext()
  );
  await Promise.all(workers);
  return results;
}

async function main() {
  const startTime = Date.now();
  const progress = loadProgress();

  // Each seed generates 10 variations by running 10 times with high temperature
  const VARIATIONS_PER_SEED = 10;
  const totalTarget = TARGETED_SEEDS.length * VARIATIONS_PER_SEED;

  console.log("\n=".repeat(60));
  console.log("  Targeted Training Data Generator (Mercury 2)");
  console.log("=".repeat(60));
  console.log(`  Model: ${MODEL}`);
  console.log(`  Seeds: ${TARGETED_SEEDS.length}`);
  console.log(`  Variations per seed: ${VARIATIONS_PER_SEED}`);
  console.log(`  Total target: ${totalTarget} examples`);
  console.log(`  Concurrency: ${MAX_CONCURRENT}`);
  console.log(`  Resuming from: ${progress.completed} examples\n`);

  let generated = progress.completed;
  let failed = 0;

  const tasks: (() => Promise<void>)[] = [];

  for (let seedIdx = 0; seedIdx < TARGETED_SEEDS.length; seedIdx++) {
    const seed = TARGETED_SEEDS[seedIdx];

    for (let v = 0; v < VARIATIONS_PER_SEED; v++) {
      const taskId = `seed${seedIdx}_v${v}`;
      if (progress.completedSeeds.includes(taskId)) continue;

      tasks.push(async () => {
        const shortSeed = seed.length > 55 ? seed.slice(0, 55) + "..." : seed;
        console.log(`[${generated}/${totalTarget}] ${shortSeed} (v${v + 1})`);

        const example = await generateOne(seed);
        if (example) {
          const mlxLine = JSON.stringify(toMLXFormat(example));
          writeFileSync(OUTPUT_FILE, mlxLine + "\n", { flag: "a" });
          generated++;
          progress.completed = generated;
          progress.completedSeeds.push(taskId);
          saveProgress(progress);
        } else {
          failed++;
        }
      });
    }
  }

  await runWithConcurrency(tasks, MAX_CONCURRENT);

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
  console.log(`\n${"=".repeat(60)}`);
  console.log(`  Complete!`);
  console.log(`  Generated: ${generated} examples`);
  console.log(`  Failed: ${failed}`);
  console.log(`  Time: ${elapsed}s`);
  console.log(`  Output: ${OUTPUT_FILE}`);
  console.log(`${"=".repeat(60)}`);
}

main().catch(console.error);
