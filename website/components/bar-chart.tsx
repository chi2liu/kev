type Series = {
  label: string;
  color: string;
};

type Group = {
  label: string;
  values: (number | null)[];
};

type BarChartProps = {
  title: string;
  series: Series[];
  groups: Group[];
  min?: number;
  max?: number;
  lowerIsBetter?: boolean;
  caption?: string;
  digits?: number;
};

export function BarChart({
  title,
  series,
  groups,
  min = 0,
  max = 1,
  lowerIsBetter = false,
  caption,
  digits = 3,
}: BarChartProps) {
  const width = (value: number) => `${Math.max(0, ((value - min) / (max - min)) * 100)}%`;

  return (
    <figure className="not-prose my-6 rounded-xl border bg-fd-card p-4 text-sm">
      <figcaption className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-medium">{title}</span>
        <span className="text-xs text-fd-muted-foreground">
          {lowerIsBetter ? "Lower is better" : "Higher is better"}
        </span>
      </figcaption>
      <div className="mb-4 flex flex-wrap gap-3 text-xs text-fd-muted-foreground">
        {series.map((s) => (
          <span key={s.label} className="inline-flex items-center gap-1.5">
            <span className="size-2.5 rounded-sm" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
      <div className="flex flex-col gap-4">
        {groups.map((group) => (
          <div key={group.label}>
            <div className="mb-1 text-xs font-medium">{group.label}</div>
            <div className="flex flex-col gap-1">
              {series.map((s, i) => {
                const value = group.values[i];
                return (
                  <div key={s.label} className="flex items-center gap-2">
                    <div className="h-4 flex-1 overflow-hidden rounded bg-fd-muted">
                      {value != null && (
                        <div
                          className="h-full rounded"
                          style={{ width: width(value), background: s.color }}
                          role="img"
                          aria-label={`${s.label}: ${value.toFixed(digits)}`}
                        />
                      )}
                    </div>
                    <span className="w-12 text-right font-mono text-xs tabular-nums">
                      {value == null ? "–" : value.toFixed(digits)}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {caption && <p className="mt-4 text-xs text-fd-muted-foreground">{caption}</p>}
    </figure>
  );
}
