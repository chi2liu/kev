export function KevLogo() {
  return (
    <span className="inline-flex items-center gap-2 font-semibold tracking-tight">
      <svg aria-hidden="true" viewBox="0 0 24 24" className="size-5 text-fd-primary">
        <rect x="2" y="2" width="20" height="20" rx="5" fill="currentColor" />
        <path
          d="M8 6.5v11M8 12l6.5-5.5M10.2 10.2 15.5 17.5"
          stroke="var(--color-fd-background)"
          strokeWidth="2.2"
          strokeLinecap="round"
          fill="none"
        />
      </svg>
      Kev
    </span>
  );
}
