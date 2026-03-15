import { readFileSync } from "fs";

const INPUT_FILE = "output/training_data.jsonl";

function main() {
  const content = readFileSync(INPUT_FILE, "utf-8");
  const lines = content.split("\n").filter((l) => l.trim().length > 0);

  let totalTools = 0;
  let totalSkills = 0;
  let totalSteps = 0;
  let totalConstraints = 0;
  let totalInputTokenEstimate = 0;
  let totalOutputTokenEstimate = 0;
  const categories = new Map<string, number>();
  const toolNames = new Map<string, number>();

  for (const line of lines) {
    const parsed = JSON.parse(line);
    const messages = parsed.messages;
    const userMsg = messages[1].content;
    const assistantMsg = messages[2].content;

    // Rough token estimate: ~4 chars per token
    totalInputTokenEstimate += Math.ceil(
      (messages[0].content.length + userMsg.length) / 4
    );
    totalOutputTokenEstimate += Math.ceil(assistantMsg.length / 4);

    const agentDef = JSON.parse(assistantMsg);
    const agent = agentDef.agent;

    totalTools += agent.tools?.length || 0;
    totalSkills += agent.skills?.length || 0;
    totalConstraints += agent.constraints?.length || 0;

    for (const skill of agent.skills || []) {
      totalSteps += skill.steps?.length || 0;
    }

    for (const tool of agent.tools || []) {
      const count = toolNames.get(tool.name) || 0;
      toolNames.set(tool.name, count + 1);
    }

    // Extract category from agent name or description
    const name = agent.name || "unknown";
    const cat = name.split("-")[0];
    categories.set(cat, (categories.get(cat) || 0) + 1);
  }

  const n = lines.length;

  console.log(`\nDataset Statistics`);
  console.log(`  Examples:           ${n}`);
  console.log(`  Avg tools/agent:    ${(totalTools / n).toFixed(1)}`);
  console.log(`  Avg skills/agent:   ${(totalSkills / n).toFixed(1)}`);
  console.log(`  Avg steps/skill:    ${(totalSteps / totalSkills).toFixed(1)}`);
  console.log(`  Avg constraints:    ${(totalConstraints / n).toFixed(1)}`);
  console.log(
    `  Est. input tokens:  ${(totalInputTokenEstimate / 1000).toFixed(0)}K`
  );
  console.log(
    `  Est. output tokens: ${(totalOutputTokenEstimate / 1000).toFixed(0)}K`
  );
  console.log(
    `  Est. total tokens:  ${((totalInputTokenEstimate + totalOutputTokenEstimate) / 1000).toFixed(0)}K`
  );

  // Top 15 most common tool names
  const sortedTools = [...toolNames.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15);
  console.log(`\nTop 15 Tool Names:`);
  for (const [name, count] of sortedTools) {
    console.log(`  ${name}: ${count}`);
  }
}

main();
