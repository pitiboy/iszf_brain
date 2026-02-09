import { createOpenAI } from "@ai-sdk/openai";
import { embedMany } from "ai";
import { env } from "@/lib/env.mjs";

const openai = createOpenAI({ apiKey: env.OPENAI_API_KEY });

const generateChunks = (input: string): string[] => {
  return input
    .trim()
    .split(".")
    .filter((i) => i !== "");
};

export const generateEmbeddings = async (
  value: string
): Promise<Array<{ embedding: number[]; content: string }>> => {
  const chunks = generateChunks(value);
  const { embeddings } = await embedMany({
    model: openai.embeddingModel("text-embedding-ada-002"),
    values: chunks,
  });
  return embeddings.map((e, i) => ({ content: chunks[i], embedding: e }));
};
