// Run once per model to create its workload: a live research agent answers the same 10 FRAMES questions
// with Wikipedia tools, and each conversation is saved to traces/<model>/ for bench.ts to replay on every
// provider that serves that model. Commit the traces.
//
// FRAMES (https://huggingface.co/datasets/google/frames-benchmark) is a set of multi-hop questions
// that need facts from several Wikipedia articles, so answering one takes several tool calls.
//
// Usage: npm run record -- <cell-id> [<cell-id> ...] [--unfinished]     e.g.  npm run record -- modal/glm-5.3 openai/gpt-6.1-sol
// --unfinished re-records only tasks whose saved trace doesn't end in a final answer.
// The provider used for recording doesn't matter: the trace belongs to the model.
import { generateText, isStepCount, type ModelMessage, type TextPart, type ToolCallPart } from 'ai';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cells, modelOf, reasoningFor } from './providers.ts';
import { INSTRUCTIONS, tools } from './tools.ts';

// The FRAMES questions (row indexes) every model answers: multi-turn questions that every model finishes
// (reaches a final answer) within MAX_TURNS.
const QUESTIONS = [0, 7, 42, 63, 77, 84, 112, 119, 140, 161];
const MAX_TURNS = 50; // high enough for hard questions to finish; a task that hits it never answered

// --- Wikipedia tools, with a committed cache ------------------------------------------------------

// Every Wikipedia response is saved in traces/wikipedia-cache.json. Models that make the same lookup get
// byte-identical results, and re-recording later doesn't depend on Wikipedia having changed.
const CACHE_FILE = 'traces/wikipedia-cache.json';
const cache: Record<string, unknown> = existsSync(CACHE_FILE) ? JSON.parse(readFileSync(CACHE_FILE, 'utf8')) : {};

// Uncached requests go one at a time: Wikimedia allows 200 requests/min for clients that identify
// themselves with a contact URL in the User-Agent (only 10/min otherwise), and asks clients not to burst.
let queue: Promise<unknown> = Promise.resolve();
async function wikipedia(params: Record<string, string>): Promise<any> {
  const url = 'https://en.wikipedia.org/w/api.php?' + new URLSearchParams({ format: 'json', formatversion: '2', ...params });
  if (!(url in cache)) {
    cache[url] = await (queue = queue.then(() => fetchWikipedia(url), () => fetchWikipedia(url)));
    writeFileSync(CACHE_FILE, JSON.stringify(cache, null, 1));
  }
  return cache[url];
}

async function fetchWikipedia(url: string) {
  const headers = { 'User-Agent': 'inference-bench/0.1 (https://github.com/powerset-co/inference-bench)' };
  for (let attempt = 1; ; attempt++) {
    const res = await fetch(url, { headers });
    if (res.ok) return res.json();
    if (attempt === 4) throw new Error(`Wikipedia returned HTTP ${res.status}`);
    await new Promise((r) => setTimeout(r, 1000 * Number(res.headers.get('retry-after') ?? 2)));
  }
}

async function search({ query }: { query: string }) {
  const data = await wikipedia({ action: 'query', list: 'search', srsearch: query, srlimit: '5' });
  return data.query.search.map((r: any) => `${r.title}: ${r.snippet.replace(/<[^>]+>/g, '')}`).join('\n') || 'No results.';
}

async function getPage({ title }: { title: string }) {
  const data = await wikipedia({ action: 'query', prop: 'extracts', explaintext: '1', redirects: '1', titles: title });
  return data.query.pages[0]?.extract?.slice(0, 20000) || 'Page not found.';
}

// --- FRAMES questions, with a committed cache ------------------------------------------------------

// The questions are saved in traces/questions.json the first time, so later recordings don't need the
// Hugging Face API (which sometimes returns an error page instead of data).
const QUESTIONS_FILE = 'traces/questions.json';
const questions: Record<number, { question: string; answer: string }> = existsSync(QUESTIONS_FILE) ? JSON.parse(readFileSync(QUESTIONS_FILE, 'utf8')) : {};

async function frames(index: number) {
  for (let attempt = 1; !questions[index]; attempt++) {
    const url = `https://datasets-server.huggingface.co/rows?dataset=google%2Fframes-benchmark&config=default&split=test&offset=${index}&length=1`;
    const res = await fetch(url);
    if (res.ok) {
      const { Prompt: question, Answer: answer } = ((await res.json()) as any).rows[0].row;
      questions[index] = { question, answer };
      writeFileSync(QUESTIONS_FILE, JSON.stringify(questions, null, 2));
    } else if (attempt === 5) throw new Error(`Hugging Face returned HTTP ${res.status}`);
    else await new Promise((r) => setTimeout(r, 2000 * attempt));
  }
  return questions[index];
}

// --- Record ---------------------------------------------------------------------------------------

const args = process.argv.slice(2);
const unfinishedOnly = args.includes('--unfinished');
const finished = (file: string) => existsSync(file) && readFileSync(file, 'utf8').includes('ANSWER:');

for (const cellId of args.filter((a) => !a.startsWith('--'))) {
  const cell = cells.find((c) => c.id === cellId);
  if (!cell) throw new Error(`Unknown cell "${cellId}". Options: ${cells.map((c) => c.id).join(', ')}`);
  const dir = `traces/${modelOf(cell.id)}`;
  mkdirSync(dir, { recursive: true });

  for (const index of QUESTIONS) {
    if (unfinishedOnly && finished(`${dir}/frames-${index}.json`)) continue;
    const { question, answer } = await frames(index);
    try {
      const result = await generateText({
        model: cell.model,
        instructions: INSTRUCTIONS,
        prompt: question,
        tools: { search_wikipedia: { ...tools.search_wikipedia, execute: search }, get_page: { ...tools.get_page, execute: getPage } },
        stopWhen: isStepCount(MAX_TURNS),
        reasoning: reasoningFor(cell.id),
      });
      const messages = [{ role: 'user', content: question }, ...neutral(result.responseMessages)];
      const trace = { id: `frames-${index}`, question, answer, recordedWith: cell.id, messages };
      writeFileSync(`${dir}/frames-${index}.json`, JSON.stringify(trace, null, 2));
      console.log(`${cell.id} frames-${index}: ${result.steps.length} turns`);
    } catch (e) {
      console.log(`${cell.id} frames-${index}: FAILED, not saved (${String(e).slice(0, 200)})`);
    }
  }
}

/**
 * Strip provider-specific details (reasoning text and signatures, metadata, tool-call ID formats) so a
 * trace recorded on one provider replays cleanly on any other.
 */
function neutral(messages: ModelMessage[]): ModelMessage[] {
  const ids = new Map<string, string>();
  const id = (x: string) => ids.get(x) ?? ids.set(x, `call_${ids.size}`).get(x)!;
  return messages.map((m): ModelMessage => {
    if (m.role === 'assistant') {
      const parts = typeof m.content === 'string' ? [{ type: 'text' as const, text: m.content }] : m.content;
      return {
        role: 'assistant',
        content: parts.flatMap<TextPart | ToolCallPart>((p) =>
          p.type === 'text' && p.text.trim() ? [{ type: 'text', text: p.text }]
          : p.type === 'tool-call' ? [{ type: 'tool-call', toolCallId: id(p.toolCallId), toolName: p.toolName, input: p.input }]
          : [],
        ),
      };
    }
    if (m.role === 'tool') {
      return {
        role: 'tool',
        content: m.content.flatMap((p) =>
          p.type === 'tool-result' ? [{ type: 'tool-result' as const, toolCallId: id(p.toolCallId), toolName: p.toolName, output: p.output }] : [],
        ),
      };
    }
    return m;
  });
}
