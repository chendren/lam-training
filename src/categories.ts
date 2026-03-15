// Each category has seed prompts that Claude will diversify and expand.
// 20 categories x 5 seeds x 10 variations per batch = 1,000 examples.

export interface Category {
  name: string;
  description: string;
  seeds: string[];
}

export const CATEGORIES: Category[] = [
  {
    name: "code-review",
    description: "Agents that review code for quality, security, style, or correctness",
    seeds: [
      "Create an agent that reviews pull requests for security vulnerabilities",
      "Build me a code reviewer that checks for OWASP top 10 issues",
      "I need an agent that enforces our team's TypeScript style guide",
      "Make an agent that catches common React anti-patterns in PRs",
      "Create a reviewer that checks for proper error handling in async code",
    ],
  },
  {
    name: "devops-automation",
    description: "Agents that automate CI/CD, deployment, and infrastructure tasks",
    seeds: [
      "Build an agent that manages blue-green deployments on AWS",
      "Create a deployment agent that runs canary releases with automatic rollback",
      "I need an agent to manage my Kubernetes cluster scaling",
      "Make an agent that monitors CloudWatch and auto-scales Lambda concurrency",
      "Build an agent that validates CloudFormation templates before deployment",
    ],
  },
  {
    name: "data-pipeline",
    description: "Agents that orchestrate data ingestion, transformation, and loading",
    seeds: [
      "Create an agent that monitors S3 for new CSV files and loads them into Postgres",
      "Build a data quality agent that validates incoming data against schemas",
      "I need an agent that deduplicates records across multiple data sources",
      "Make an agent that transforms JSON API responses into normalized database rows",
      "Create an ETL agent that handles incremental data syncs between systems",
    ],
  },
  {
    name: "testing",
    description: "Agents that generate, run, or manage tests",
    seeds: [
      "Build an agent that generates unit tests for new functions automatically",
      "Create an agent that runs regression tests and reports flaky test patterns",
      "I need an agent that generates API integration tests from OpenAPI specs",
      "Make an agent that monitors test coverage and creates tests for uncovered paths",
      "Build an agent that generates load test scenarios from production traffic patterns",
    ],
  },
  {
    name: "documentation",
    description: "Agents that create, update, or validate documentation",
    seeds: [
      "Create an agent that generates API docs from code comments and types",
      "Build an agent that keeps README files in sync with actual project setup steps",
      "I need an agent that generates changelog entries from git commits",
      "Make an agent that creates architecture decision records from design discussions",
      "Build an agent that validates code examples in documentation actually compile",
    ],
  },
  {
    name: "security",
    description: "Agents that scan, monitor, or enforce security policies",
    seeds: [
      "Create an agent that scans dependencies for known vulnerabilities daily",
      "Build a secrets scanner that checks git history for leaked credentials",
      "I need an agent that enforces least-privilege IAM policies in AWS",
      "Make an agent that monitors authentication logs for suspicious patterns",
      "Create an agent that validates CORS and CSP headers on all endpoints",
    ],
  },
  {
    name: "monitoring-alerting",
    description: "Agents that watch systems and respond to incidents",
    seeds: [
      "Build an agent that monitors API latency and auto-creates incident tickets",
      "Create an agent that correlates errors across microservices to find root causes",
      "I need an agent that watches database query performance and suggests index changes",
      "Make an agent that detects anomalies in user signup rates",
      "Build an on-call agent that triages alerts and routes to the right team",
    ],
  },
  {
    name: "customer-support",
    description: "Agents that handle customer interactions and support workflows",
    seeds: [
      "Create a support agent that categorizes incoming tickets by urgency and topic",
      "Build an agent that drafts responses to common customer questions",
      "I need an agent that escalates tickets when sentiment turns negative",
      "Make an agent that tracks SLA compliance and alerts before breaches",
      "Create an agent that generates customer-facing status updates during incidents",
    ],
  },
  {
    name: "content-generation",
    description: "Agents that create, edit, or manage content",
    seeds: [
      "Build an agent that generates social media posts from blog articles",
      "Create an agent that summarizes meeting transcripts into action items",
      "I need an agent that localizes marketing copy for different regions",
      "Make an agent that generates product descriptions from feature specs",
      "Create an agent that drafts weekly team status reports from project updates",
    ],
  },
  {
    name: "database-management",
    description: "Agents that manage database operations, migrations, and optimization",
    seeds: [
      "Create an agent that generates migration scripts from schema diffs",
      "Build an agent that monitors slow queries and suggests optimizations",
      "I need an agent that manages database backup schedules and validates restores",
      "Make an agent that detects schema drift between environments",
      "Build an agent that generates seed data for development environments",
    ],
  },
  {
    name: "project-management",
    description: "Agents that track tasks, plan sprints, or manage workflows",
    seeds: [
      "Create an agent that breaks down user stories into technical tasks",
      "Build an agent that detects blocked tasks and suggests unblocking actions",
      "I need an agent that estimates sprint velocity from historical data",
      "Make an agent that generates standup summaries from git and ticket activity",
      "Create an agent that identifies scope creep by comparing PRs to original tickets",
    ],
  },
  {
    name: "api-management",
    description: "Agents that design, validate, or manage APIs",
    seeds: [
      "Build an agent that validates API requests against OpenAPI schemas",
      "Create an agent that detects breaking changes between API versions",
      "I need an agent that generates API client SDKs from OpenAPI specs",
      "Make an agent that monitors API rate limits and implements backpressure",
      "Create an agent that generates mock API servers from endpoint definitions",
    ],
  },
  {
    name: "ml-ops",
    description: "Agents that manage ML model lifecycle, training, and deployment",
    seeds: [
      "Create an agent that monitors model drift and triggers retraining",
      "Build an agent that manages A/B testing between model versions",
      "I need an agent that tracks experiment metrics and ranks model candidates",
      "Make an agent that validates training data quality before pipeline runs",
      "Build an agent that optimizes hyperparameters across training runs",
    ],
  },
  {
    name: "compliance",
    description: "Agents that enforce regulatory or policy compliance",
    seeds: [
      "Create an agent that scans code for GDPR data handling violations",
      "Build an agent that generates SOC2 compliance evidence from system logs",
      "I need an agent that validates data retention policies are properly enforced",
      "Make an agent that checks accessibility compliance on web pages",
      "Create an agent that audits user permissions against role definitions",
    ],
  },
  {
    name: "communication",
    description: "Agents that manage notifications, messages, or cross-team coordination",
    seeds: [
      "Build an agent that routes Slack messages to the right channel based on content",
      "Create an agent that sends deployment notifications to stakeholders",
      "I need an agent that digests email threads and extracts decisions",
      "Make an agent that coordinates handoffs between time-zone-distributed teams",
      "Create an agent that generates release announcements from changelogs",
    ],
  },
  {
    name: "cost-optimization",
    description: "Agents that monitor and optimize cloud or operational costs",
    seeds: [
      "Create an agent that identifies unused AWS resources and recommends cleanup",
      "Build an agent that tracks cloud spend trends and forecasts budget overruns",
      "I need an agent that right-sizes EC2 instances based on utilization data",
      "Make an agent that compares reserved vs on-demand pricing for our workloads",
      "Create an agent that detects cost anomalies in daily cloud billing",
    ],
  },
  {
    name: "onboarding",
    description: "Agents that help onboard new team members or users",
    seeds: [
      "Build an agent that creates personalized onboarding checklists for new developers",
      "Create an agent that answers questions about codebase architecture for new hires",
      "I need an agent that sets up development environments automatically",
      "Make an agent that pairs new team members with relevant code reviewers",
      "Create an agent that generates role-specific learning paths from team documentation",
    ],
  },
  {
    name: "research",
    description: "Agents that gather, analyze, or synthesize information",
    seeds: [
      "Create an agent that monitors competitor product launches and summarizes changes",
      "Build an agent that searches academic papers for relevant techniques",
      "I need an agent that aggregates user feedback from multiple channels into themes",
      "Make an agent that benchmarks our API performance against industry standards",
      "Create an agent that tracks technology trends relevant to our stack",
    ],
  },
  {
    name: "workflow-automation",
    description: "Agents that automate repetitive business processes",
    seeds: [
      "Build an agent that processes invoice PDFs and enters data into our accounting system",
      "Create an agent that automates employee time-off request approvals",
      "I need an agent that generates weekly reports from multiple data sources",
      "Make an agent that manages document approval workflows with multi-step sign-off",
      "Create an agent that reconciles transactions between two financial systems",
    ],
  },
  {
    name: "multi-agent-orchestration",
    description: "Systems of multiple agents working together",
    seeds: [
      "Create a system where a planner agent breaks down tasks and assigns them to specialist agents",
      "Build an agent team: one researches, one writes, one reviews",
      "I need an orchestrator that coordinates between a data agent and a visualization agent",
      "Make a pipeline of agents: intake, validate, transform, load, verify",
      "Create a debate system where two agents argue for and against a technical decision",
    ],
  },
];
