import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import { KevLogo } from "@/components/kev-logo";
import { links } from "./shared";

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: <KevLogo />,
    },
    githubUrl: links.github,
    links: [
      { text: "Docs", url: "/docs", active: "nested-url" },
      { text: "Benchmarks", url: "/docs/benchmarks", active: "nested-url" },
      { text: "Architecture", url: "/docs/architecture", active: "nested-url" },
      { text: "Models", url: links.huggingFace, external: true },
    ],
  };
}
