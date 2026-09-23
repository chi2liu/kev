import { choice, noul, score, TypeSafeClient } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({
  apiKey: process.env.KEV_API_KEY ?? "local",
  baseURL: process.env.KEV_URL ?? "http://127.0.0.1:8009",
  defaultModel: "kev-latest",
});

const { answers } = await client.systemOne({
  state: "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  questions: {
    department: choice("Which team should handle this?", {
      returns: "Exchanges, refunds, wrong or damaged items",
      shipping: "Delivery status, delays, lost packages",
      billing: "Charges, invoices, payment problems",
    }),
    escalate: noul("Does this need urgent human attention?"),
    frustration: score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
  },
});

// Every field is typed from the questions you asked.
console.log(answers.department.choice); // "returns" | "shipping" | "billing"
console.log(answers.department.probabilities.billing); // number between 0 and 1
console.log(answers.escalate.noul); // probability of "yes"
console.log(answers.frustration.score); // 0 = Calm … 2 = Very angry, can be fractional
