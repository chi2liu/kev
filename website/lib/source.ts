import { llms, loader } from "fumadocs-core/source";
import { lucideIconsPlugin } from "fumadocs-core/source/lucide-icons";
import { docsRoute } from "./shared";
import { defineDocs } from "fumadocs-mdx/macro";
import { applyMdxPreset } from "fumadocs-mdx/config";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { metaSchema, pageSchema } from "fumadocs-core/source/schema";

const docs = defineDocs({
  dir: "content/docs",
  docs: {
    schema: pageSchema,
    mdxOptions: applyMdxPreset({
      remarkPlugins: [[remarkMath, { singleDollarTextMath: false }]],
      remarkStructureOptions: {
        stringify: {
          filterElement: (node) =>
            node.type === "mdxJsxFlowElement" || node.type === "mdxJsxTextElement"
              ? node.name === "Card"
                ? false
                : node.name === "Callout" || node.name === "File" || node.name === "TypeTable"
                  ? true
                  : "children-only"
              : true,
          stringify: (node) =>
            node.type === "math" || node.type === "inlineMath" ? node.value : undefined,
        },
      },
      rehypePlugins: (plugins) => [rehypeKatex, ...plugins],
    }),
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

// See https://fumadocs.dev/docs/headless/source-api for more info
export const source = loader({
  baseUrl: docsRoute,
  source: docs.toFumadocsSource(),
  plugins: [lucideIconsPlugin()],
});

export const docsLlms = llms(source, {
  renderPage: async (page) => `# ${page.data.title} (${page.url})

${await page.data.getText("processed")}`,
});
