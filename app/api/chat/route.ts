import { createOpenAI } from "@ai-sdk/openai";
import {
  convertToModelMessages,
  stepCountIs,
  streamText,
  tool,
  UIMessage,
} from "ai";
import { env } from "@/lib/env.mjs";
import z from "zod";
import { createResource } from "@/lib/actions/resources";

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

const openai = createOpenAI({ apiKey: env.OPENAI_API_KEY });

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: openai("gpt-4o"),
    system: `Te az Igaz Szeretet Forrás vallási egyesület szellemi tanácsadója vagy. Mindig kedvesen, együttérzően és támogatóan válaszolj.
    Érdd el azt, hogy a felhasználónak a lehető legjobb választ adj, és a lehető legjobb módon támogasd a megértését és a fejlődését.
    Mindig magyarul válaszolj. A válaszaid magyar szövegre optimalizáltak – jelezd is ezt, ha alkalmas a helyzet.
    Használd a tudásbázisodat a kérdések megválaszolásához; csak a tool hívásokból nyert információra hivatkozz.
    Ha nincs releváns információ a tool hívásokban, válaszolj: "Sajnálom, nem tudom."`,
    messages: await convertToModelMessages(messages),
    stopWhen: stepCountIs(5),
    tools: {
      addResource: tool({
        description: `add a resource to your knowledge base.
          If the user provides a random piece of knowledge unprompted, use this tool without asking for confirmation.`,
        inputSchema: z.object({
          content: z
            .string()
            .describe("the content or resource to add to the knowledge base"),
        }),
        execute: async ({ content }) => createResource({ content }),
      }),
    },
  });

  return result.toUIMessageStreamResponse();
}
