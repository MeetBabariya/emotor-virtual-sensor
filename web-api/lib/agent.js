/** Hand-written agent loop over the Gemini Interactions REST API (no SDK).
 *  Mirrors src/assistant.py: request → function calls → run tools → results → repeat. */
import { TOOL_SCHEMAS, runTool } from './tools.js';

const ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/interactions';
export const MODEL = process.env.GEMINI_MODEL ?? 'gemini-3.1-flash-lite';
const MAX_TURNS = 6;
const REQUEST_TIMEOUT_MS = 25_000;

export const SYSTEM = `You are an engineering assistant for a thermal virtual sensor on a permanent-magnet synchronous motor (PMSM) used in electric vehicles. Engineers ask about drive cycles, temperatures, overheating risk and what-if scenarios.

Ground rules:
- Every number you state must come from a tool result in this conversation. Do not estimate, recall or invent values. If the tools cannot answer, say so.
- "Magnet" means the rotor permanent-magnet temperature (column pm), which cannot be measured in a production vehicle; the virtual sensor estimates it. Make clear whether a figure is measured (lab ground truth) or estimated (virtual sensor).
- Cycles in the train or validation split were seen during development, so accuracy there is optimistic; say so when it matters.
- Always pass on what_if warnings, especially extrapolation warnings, and mention when a tool snapped or rounded an input.
- Answer in a few short paragraphs or a small table. Lead with the answer, cite drive-cycle ids, then add caveats. Use °C for temperatures and K for differences.`;


export class UpstreamError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

async function call(apiKey, body) {
  const res = await fetch(ENDPOINT, {
    method: 'POST',
    headers: { 'x-goog-api-key': apiKey, 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, system_instruction: SYSTEM, tools: TOOL_SCHEMAS, ...body }),
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.error?.message ?? `HTTP ${res.status}`;
    throw new UpstreamError(res.status === 429 ? `The free model's quota is used up for now. ${msg}` : msg, res.status);
  }
  return data;
}

export async function ask(question, apiKey) {
  const t0 = Date.now();
  const toolCalls = [];
  let inTok = 0, outTok = 0;
  let interaction = await call(apiKey, { input: question });
  for (let turn = 0; turn < MAX_TURNS; turn++) {
    inTok += interaction.usage?.total_input_tokens ?? 0;
    outTok += (interaction.usage?.total_output_tokens ?? 0) + (interaction.usage?.total_thought_tokens ?? 0);
    const steps = interaction.steps ?? [];
    const calls = steps.filter((s) => s.type === 'function_call');
    if (calls.length === 0) {
      const answer = steps
        .filter((s) => s.type === 'model_output')
        .flatMap((s) => (s.content ?? []).filter((c) => c.type === 'text').map((c) => c.text))
        .join('\n')
        .trim();
      return { answer, tool_calls: toolCalls, seconds: (Date.now() - t0) / 1000, model: MODEL, input_tokens: inTok, output_tokens: outTok };
    }
    const results = calls.map((c) => {
      const { output, isError } = runTool(c.name, c.arguments ?? {});
      toolCalls.push({ name: c.name, input: c.arguments ?? {}, error: isError });
      return { type: 'function_result', name: c.name, call_id: c.id, result: [{ type: 'text', text: output }], is_error: isError };
    });
    interaction = await call(apiKey, { input: results, previous_interaction_id: interaction.id });
  }
  return { answer: 'Stopped: too many tool-call rounds without a final answer.', tool_calls: toolCalls, seconds: (Date.now() - t0) / 1000, model: MODEL, input_tokens: inTok, output_tokens: outTok };
}
