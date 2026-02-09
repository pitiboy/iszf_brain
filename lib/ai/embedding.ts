import { createOpenAI } from "@ai-sdk/openai";
import { embed, embedMany } from "ai";
import { db } from "../db";
import { cosineDistance, desc, gt, sql } from "drizzle-orm";
import { embeddings } from "../db/schema/embeddings";
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
  console.debug("generateEmbeddings value", value);
  const chunks = generateChunks(value);
  const { embeddings } = await embedMany({
    model: openai.embeddingModel("text-embedding-ada-002"),
    values: chunks,
  });
  return embeddings.map((e, i) => ({ content: chunks[i], embedding: e }));
};

export const generateEmbedding = async (value: string): Promise<number[]> => {
  const input = value.replaceAll("\\n", " ");
  const { embedding } = await embed({
    model: openai.embeddingModel("text-embedding-ada-002"),
    value: input,
  });
  return embedding;
};

export const findRelevantContent = async (userQuery: string) => {
  const userQueryEmbedded = await generateEmbedding(userQuery);
  console.debug("searching database for userQuery", userQuery);
  const distance = cosineDistance(
    embeddings.embedding as any,
    userQueryEmbedded
  );
  const similarity = sql<number>`1 - (${distance})`.as("similarity");
  const similarityThreshold = 0.3;
  const similarGuides = await db
    .select({ name: embeddings.content, similarity })
    .from(embeddings)
    .where(gt(sql`1 - (${distance})`, similarityThreshold))
    .orderBy(desc(similarity))
    .limit(4);
  return similarGuides;
};
