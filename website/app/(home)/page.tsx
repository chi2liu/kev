import Link from "next/link";
import { links } from "@/lib/shared";

const request = `// State: "Shoes arrived two weeks late and in the wrong
//         size. Also I see two charges on my card."
// Response from Kev-4B:
{
  "department": {
    "type": "choice", "choice": "returns", "confidence": 0.21,
    "probabilities": { "returns": 0.47, "shipping": 0.28,
                       "billing": 0.25 }
  },
  "escalate": { "type": "noul", "noul": 0.93 },
  "frustration": {
    "type": "score", "score": 1.44, "confidence": 0.78,
    "legend": { "0": "Calm", "1": "Frustrated",
                "2": "Very angry" },
    "probabilities": { "0": 0.00, "1": 0.56, "2": 0.44 }
  }
}`;

const features = [
  {
    title: "Answers, not text",
    body: "Give Kev a document and typed questions. It returns a choice, a score or a yes/no probability for each. It cannot ramble, invent a label, or break your JSON parser.",
  },
  {
    title: "Probabilities you can act on",
    body: "Every answer comes with a calibrated probability, so your app can act on sure answers and send unsure ones to a person.",
  },
  {
    title: "Drop-in for the TypeSafe SDK",
    body: "Kev speaks the TypeSafe System One API. Change the base URL, keep your code. Works with the Vercel AI SDK's experimental_evaluate() too.",
  },
  {
    title: "Yours to run",
    body: "Apache-2.0 weights at 0.8B, 4B and 9B parameters. Run on a laptop, or deploy to a Modal GPU with one command that scales to zero.",
  },
];

export default function HomePage() {
  return (
    <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-16 px-6 py-16">
      <section className="grid items-center gap-10 lg:grid-cols-2">
        <div className="flex flex-col gap-6">
          <p className="text-sm font-medium text-fd-primary">Open decision models</p>
          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
            Ask your data typed questions. Get calibrated answers.
          </h1>
          <p className="text-lg text-fd-muted-foreground">
            Kev reads a support ticket, a form, a log line or a chess board and answers the
            questions your product needs, like &ldquo;is this about billing?&rdquo; or &ldquo;how
            urgent is it?&rdquo;, with a probability for every answer. No prompt engineering, no
            output parsing, no AI background required.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link
              href="/docs/quickstart"
              className="rounded-lg bg-fd-primary px-4 py-2 text-sm font-medium text-fd-primary-foreground"
            >
              Quickstart
            </Link>
            <Link href="/docs" className="rounded-lg border px-4 py-2 text-sm font-medium">
              What is Kev?
            </Link>
            <Link
              href="/docs/deploy/modal"
              className="rounded-lg border px-4 py-2 text-sm font-medium"
            >
              Deploy on Modal
            </Link>
          </div>
        </div>
        <pre className="overflow-x-auto rounded-xl border bg-fd-card p-5 text-[13px] leading-relaxed">
          <code>{request}</code>
        </pre>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {features.map((f) => (
          <div key={f.title} className="rounded-xl border bg-fd-card p-5">
            <h2 className="mb-2 font-medium">{f.title}</h2>
            <p className="text-sm text-fd-muted-foreground">{f.body}</p>
          </div>
        ))}
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <Link href="/docs/benchmarks" className="rounded-xl border p-5 hover:bg-fd-accent">
          <h2 className="mb-2 font-medium">Benchmarks</h2>
          <p className="text-sm text-fd-muted-foreground">
            Accuracy and calibration for every model size next to Jev, with the caveats that come
            with them.
          </p>
        </Link>
        <Link
          href="/docs/benchmarks/kev-vs-jev"
          className="rounded-xl border p-5 hover:bg-fd-accent"
        >
          <h2 className="mb-2 font-medium">Kev vs Jev</h2>
          <p className="text-sm text-fd-muted-foreground">
            Same API, different trade-offs: hosted and stronger, or open and yours.
          </p>
        </Link>
        <Link href="/docs/architecture" className="rounded-xl border p-5 hover:bg-fd-accent">
          <h2 className="mb-2 font-medium">Architecture</h2>
          <p className="text-sm text-fd-muted-foreground">
            For researchers: the block-causal mask, hybrid DeltaNet rows, the pointer head,
            calibration and the evaluation protocol.
          </p>
        </Link>
      </section>

      <p className="text-center text-sm text-fd-muted-foreground">
        Weights on{" "}
        <a href={links.huggingFace} className="underline">
          Hugging Face
        </a>
        , code on{" "}
        <a href={links.github} className="underline">
          GitHub
        </a>
        , and a live demo in the{" "}
        <a href={links.space} className="underline">
          Kev Space
        </a>
        .
      </p>
    </main>
  );
}
