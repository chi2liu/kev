import type {
  Experimental_EvaluationModelV4 as EvaluationModelV4,
  Experimental_EvaluationModelV4Answer as EvaluationModelV4Answer,
  Experimental_EvaluationModelV4Question as EvaluationModelV4Question,
} from "@ai-sdk/provider";

type KevSettings = {
  /** Your Kev endpoint, e.g. https://<workspace>--kev-api.modal.run or http://127.0.0.1:8009 */
  baseURL: string;
  /** The bearer key set with KEV_API_KEY. Omit for an open local server. */
  apiKey?: string;
  /** The model name Kev reports under; "kev-latest" is always served. */
  modelId?: string;
};

type KevAnswer =
  | { type: "noul"; noul: number }
  | { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> }
  | { type: "score"; score: number; confidence: number; probabilities: Record<string, number> };

type KevResponse = {
  model: string;
  answers: Record<string, KevAnswer>;
  usage?: { input_tokens: number; output_tokens: number };
};

// Kev's native yes/no type is "noul"; the AI SDK calls it "boolean".
function toKevQuestion(question: EvaluationModelV4Question) {
  return question.type === "boolean" ? { ...question, type: "noul" as const } : question;
}

function toAnswer(answer: KevAnswer): EvaluationModelV4Answer {
  switch (answer.type) {
    case "noul":
      return { type: "boolean", probability: answer.noul };
    case "choice":
      return { type: "choice", choice: answer.choice, probabilities: answer.probabilities };
    case "score":
      return { type: "score", score: answer.score, probabilities: answer.probabilities };
  }
}

export function kev({ baseURL, apiKey, modelId = "kev-latest" }: KevSettings): EvaluationModelV4 {
  return {
    specificationVersion: "v4",
    provider: "kev",
    modelId,
    supportedQuestionTypes: ["choice", "score", "boolean"],
    async doEvaluate({ state, questions, abortSignal, headers }) {
      const response = await fetch(`${baseURL.replace(/\/$/, "")}/v1/systemone`, {
        method: "POST",
        signal: abortSignal,
        headers: {
          "content-type": "application/json",
          ...(apiKey ? { authorization: `Bearer ${apiKey}` } : {}),
          ...Object.fromEntries(
            Object.entries(headers ?? {}).filter(
              (entry): entry is [string, string] => entry[1] !== undefined,
            ),
          ),
        },
        body: JSON.stringify({
          model: modelId,
          state,
          questions: Object.fromEntries(
            Object.entries(questions).map(([id, q]) => [id, toKevQuestion(q)]),
          ),
        }),
      });
      if (!response.ok) {
        throw new Error(`Kev returned ${response.status}: ${await response.text()}`);
      }
      const body = (await response.json()) as KevResponse;

      return {
        answers: Object.fromEntries(
          Object.entries(body.answers).map(([id, answer]) => [id, toAnswer(answer)]),
        ),
        // Kev rounds probabilities and scores to four decimals.
        rounding: { probabilityDecimals: 4, scoreDecimals: 4 },
        usage: {
          inputTokens: body.usage?.input_tokens,
          outputTokens: body.usage?.output_tokens,
        },
        warnings: [],
        providerMetadata: {
          kev: {
            confidence: Object.fromEntries(
              Object.entries(body.answers).flatMap(([id, a]) =>
                a.type === "noul" ? [] : [[id, a.confidence]],
              ),
            ),
          },
        },
        response: {
          id: response.headers.get("x-typesafe-request-id") ?? undefined,
          modelId: body.model,
        },
      };
    },
  };
}
