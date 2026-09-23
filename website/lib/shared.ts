import { createGetUrl } from "fumadocs-core/source";

export const appName = "Kev";
export const appDescription =
  "Open decision models that answer typed questions about your data with calibrated probabilities. Self-host them, or point the TypeSafe SDK at them.";
export const docsRoute = "/docs";
export const docsImageRoute = "/og/docs";
export const docsContentRoute = "/llms.mdx/docs";

export const gitConfig = {
  user: "jaredpalmer",
  repo: "kev",
  branch: "main",
  contentDir: "website/content/docs",
};

export const links = {
  github: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  huggingFace: "https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd",
  space: "https://huggingface.co/spaces/jaredpalmer/kev",
};

export function repoFileUrl(path: string) {
  return `${links.github}/blob/${gitConfig.branch}/${path}`;
}

const getContentUrl = createGetUrl(docsContentRoute);

export function getPageMarkdownUrl(page: { slugs: string[]; locale?: string }) {
  const segments = [...page.slugs, "content.md"];

  return { segments, url: getContentUrl(segments, page.locale) };
}

const getImageUrl = createGetUrl(docsImageRoute);

export function getPageImageUrl(page: { slugs: string[]; locale?: string }) {
  const segments = [...page.slugs, "image.png"];

  return { segments, url: getImageUrl(segments, page.locale) };
}
