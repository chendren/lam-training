import { readFileSync } from "fs";
import { TrainingExampleSchema } from "./schema.js";

const INPUT_FILE = "output/training_data.jsonl";

function main() {
  const content = readFileSync(INPUT_FILE, "utf-8");
  const lines = content.split("\n").filter((l) => l.trim().length > 0);

  let valid = 0;
  let invalid = 0;
  const errors: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    try {
      const parsed = JSON.parse(lines[i]);
      const messages = parsed.messages;

      if (!messages || messages.length !== 3) {
        errors.push(`Line ${i + 1}: Expected 3 messages, got ${messages?.length}`);
        invalid++;
        continue;
      }

      // Validate the assistant response parses back into our schema
      const assistantContent = JSON.parse(messages[2].content);
      const result = TrainingExampleSchema.safeParse({
        user_request: messages[1].content,
        ...assistantContent,
      });

      if (result.success) {
        valid++;
      } else {
        const issue = result.error.issues[0];
        errors.push(`Line ${i + 1}: ${issue?.path.join(".")} - ${issue?.message}`);
        invalid++;
      }
    } catch (err) {
      errors.push(`Line ${i + 1}: Parse error - ${err instanceof Error ? err.message : err}`);
      invalid++;
    }
  }

  console.log(`\nValidation Results`);
  console.log(`  Total:   ${lines.length}`);
  console.log(`  Valid:   ${valid}`);
  console.log(`  Invalid: ${invalid}`);
  console.log(`  Rate:    ${((valid / lines.length) * 100).toFixed(1)}%`);

  if (errors.length > 0) {
    console.log(`\nFirst 10 errors:`);
    errors.slice(0, 10).forEach((e) => console.log(`  ${e}`));
  }
}

main();
