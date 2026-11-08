// The agent's instructions and tool definitions, shared by both scripts: record.ts attaches real
// Wikipedia implementations to the tools, while bench.ts sends only the definitions.
import { tool } from 'ai';
import { z } from 'zod';

export const INSTRUCTIONS = `You are a research agent. Answer the user's question using Wikipedia.
Use search_wikipedia to find pages and get_page to read them, one tool call at a time.
Do not answer from memory: verify each fact you rely on with the tools.
When you are done, reply with a brief explanation and a final line "ANSWER: <answer>".`;

export const tools = {
  search_wikipedia: tool({
    description: 'Search English Wikipedia. Returns the top 5 page titles with snippets.',
    inputSchema: z.object({ query: z.string() }),
  }),
  get_page: tool({
    description: 'Read a Wikipedia page as plain text (the first ~20,000 characters).',
    inputSchema: z.object({ title: z.string() }),
  }),
};
