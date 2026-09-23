"use client";

import { useState } from "react";

function choiceConfidence(top: number, options: number) {
  const chance = 1 / options;
  return Math.max(0, (top - chance) / (1 - chance));
}

export function ConfidenceExplorer() {
  const [options, setOptions] = useState(3);
  const [top, setTop] = useState(0.8);
  const [threshold, setThreshold] = useState(0.7);

  const chance = 1 / options;
  const topProbability = Math.max(top, chance);
  const confidence = choiceConfidence(topProbability, options);
  const automated = confidence >= threshold;

  return (
    <div className="not-prose my-6 rounded-xl border bg-fd-card p-4 text-sm">
      <div className="mb-4 font-medium">
        Try it: from probability to confidence to a routing decision
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-fd-muted-foreground">Number of options: {options}</span>
          <input
            type="range"
            min={2}
            max={10}
            step={1}
            value={options}
            onChange={(e) => setOptions(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-xs text-fd-muted-foreground">
            Probability of the top option: {topProbability.toFixed(2)}
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={top}
            onChange={(e) => setTop(Number(e.target.value))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-xs text-fd-muted-foreground">
            Your automation threshold: {threshold.toFixed(2)}
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
        </label>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <Stat label="Random guessing would give" value={chance.toFixed(2)} />
        <Stat label="Confidence" value={confidence.toFixed(2)} />
        <div
          className={`rounded-lg border p-3 ${automated ? "border-green-500/40 bg-green-500/10" : "border-amber-500/40 bg-amber-500/10"}`}
        >
          <div className="text-xs text-fd-muted-foreground">Your app would</div>
          <div className="font-medium">{automated ? "Act automatically" : "Send to a person"}</div>
        </div>
      </div>
      <p className="mt-4 text-xs text-fd-muted-foreground">
        Confidence = (top probability − 1/options) ÷ (1 − 1/options). It is 0 when Kev is no better
        than a coin flip across the options and 1 when all probability sits on one option.
      </p>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-3">
      <div className="text-xs text-fd-muted-foreground">{label}</div>
      <div className="font-mono text-lg tabular-nums">{value}</div>
    </div>
  );
}
