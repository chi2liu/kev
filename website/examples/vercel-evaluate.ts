import { experimental_evaluate as evaluate } from "ai";
import { kev } from "./kev-evaluation-model";

const model = kev({
  baseURL: process.env.KEV_URL ?? "http://127.0.0.1:8009",
  apiKey: process.env.KEV_API_KEY,
});

const { answers } = await evaluate({
  model,
  state: {
    subject: "Charged twice",
    body: "I see two charges for order #4411. Please refund one.",
  },
  questions: {
    team: {
      type: "choice",
      instructions: "Which team should handle this ticket?",
      criteria: {
        billing: "Payments and refunds",
        shipping: "Delivery problems",
        access: "Login and account access",
      },
    },
    // Kev calls this type "noul". The AI SDK calls it "boolean".
    angry: { type: "boolean", instructions: "Is the customer angry?" },
    priority: {
      type: "score",
      instructions: "How urgent is this ticket?",
      criteria: ["low", "normal", "high"],
    },
  },
  abortSignal: AbortSignal.timeout(30_000),
});

console.log(answers.team.choice); // "billing" | "shipping" | "access"
console.log(answers.angry.probability); // P(true), between 0 and 1
console.log(answers.priority.score); // 0 = low … 2 = high
