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
import { findRelevantContent } from "@/lib/ai/embedding";

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
    Ha a felhasználó bármilyen kérdést feltehet, ELŐSZÖR mindig hívd meg a getInformation eszközt a tudásbázisban való kereséshez – mielőtt azt mondanád, hogy nem tudod. A tudásbázisban korábban megadott személyes vagy részletes más információ is lehet. 
    Próbáld meg az adatbázisban elérhető összes releváns információt kinyerni a tudásbázisból, ami a felhasználó kérdéséhez kapcsolódik.
    Válaszaid lehetőleg rövid és tömör formában legyenek.
    Csak a tool hívásokból nyert információra hivatkozz. Ha a getInformation hívás után nincs releváns eredmény, válaszolj: "Sajnálom, nem tudom."`,
    messages: await convertToModelMessages(messages),
    // stopWhen: stepCountIs(5),
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
      getInformation: tool({
        description: `Search your knowledge base for any question. ALWAYS use this tool first when the user asks a question – the knowledge base may contain personal info (eye color, preferences, etc.) the user previously stored. Only say "I don't know" after checking.`,
        inputSchema: z.object({
          question: z.string().describe("the users question"),
        }),
        execute: async ({ question }) => findRelevantContent(question),
      }),
    },
  });

  return result.toUIMessageStreamResponse();
}
