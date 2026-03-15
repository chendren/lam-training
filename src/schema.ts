import { z } from "zod";

// --- Core Output Schema: What the LAM learns to generate ---

export const ToolParameterSchema = z.object({
  name: z.string(),
  type: z.enum(["string", "number", "boolean", "array", "object"]),
  description: z.string(),
  required: z.boolean(),
});

export const ToolSchema = z.object({
  name: z.string(),
  description: z.string(),
  parameters: z.array(ToolParameterSchema),
  returns: z.string(),
});

export const SkillStepSchema = z.object({
  action: z.string(),
  tool: z.string().optional(),
  input: z.record(z.string()).optional(),
  on_failure: z.enum(["stop", "skip", "retry"]).optional(),
});

export const SkillSchema = z.object({
  name: z.string(),
  description: z.string(),
  trigger: z.string(),
  inputs: z.array(ToolParameterSchema),
  steps: z.array(SkillStepSchema),
  output: z.string(),
});

export const AgentSchema = z.object({
  name: z.string(),
  description: z.string(),
  role: z.string(),
  tools: z.array(ToolSchema),
  skills: z.array(SkillSchema),
  constraints: z.array(z.string()),
});

// --- Training Example Schema ---

export const TrainingExampleSchema = z.object({
  user_request: z.string(),
  reasoning: z.string(),
  agent: AgentSchema,
});

// --- MLX Chat Format ---

export interface MLXChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface MLXTrainingExample {
  messages: MLXChatMessage[];
}

// --- Types ---

export type ToolParameter = z.infer<typeof ToolParameterSchema>;
export type Tool = z.infer<typeof ToolSchema>;
export type SkillStep = z.infer<typeof SkillStepSchema>;
export type Skill = z.infer<typeof SkillSchema>;
export type Agent = z.infer<typeof AgentSchema>;
export type TrainingExample = z.infer<typeof TrainingExampleSchema>;

// System prompt the fine-tuned model will use at inference time
export const LAM_SYSTEM_PROMPT = `You are a Large Action Model that creates AI agents and skills from user requests.

When given a request, you:
1. Reason about what agent architecture best serves the need
2. Define the tools the agent requires
3. Define skills as composable, multi-step workflows
4. Set constraints to keep the agent safe and focused

Respond with a JSON object containing:
- reasoning: your thought process for the design
- agent: the complete agent definition with name, description, role, tools, skills, and constraints`;
