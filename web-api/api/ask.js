/** POST /api/ask  { "question": "..." }  →  { answer, tool_calls, seconds, model }
 *
 * Guards for a public endpoint on a free-tier key:
 *  - only browser requests from the portfolio origins (CORS + origin check)
 *  - question length cap
 *  - per-IP and per-instance rate limits (in-memory; best effort on serverless)
 */
import { ask, UpstreamError, MODEL, MODELS } from '../lib/agent.js';

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS ?? 'https://meetbabariya.github.io,http://localhost:4322')
  .split(',').map((s) => s.trim());
const MAX_QUESTION_CHARS = 400;
const PER_IP = { limit: 6, windowMs: 10 * 60_000 };
const PER_INSTANCE_DAILY = 150;

const hits = new Map();
let day = new Date().toISOString().slice(0, 10);
let dayCount = 0;

function setCors(req, res) {
  const origin = req.headers.origin;
  res.setHeader('access-control-allow-origin', origin && ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0]);
  res.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS');
  res.setHeader('access-control-allow-headers', 'content-type');  // x-eval-token is server-to-server only, never from browsers
  res.setHeader('vary', 'Origin');
}

function rateLimited(ip) {
  const today = new Date().toISOString().slice(0, 10);
  if (today !== day) { day = today; dayCount = 0; }
  if (dayCount >= PER_INSTANCE_DAILY) return true;
  const now = Date.now();
  const recent = (hits.get(ip) ?? []).filter((t) => now - t < PER_IP.windowMs);
  if (recent.length >= PER_IP.limit) return true;
  recent.push(now);
  hits.set(ip, recent);
  dayCount++;
  return false;
}

async function readJson(req) {
  if (req.body && typeof req.body === 'object') return req.body; // Vercel parses JSON bodies
  if (typeof req.body === 'string') return JSON.parse(req.body);
  let raw = '';
  for await (const chunk of req) raw += chunk;
  return JSON.parse(raw || '{}');
}

export default async function handler(req, res) {
  setCors(req, res);
  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method === 'GET') return res.status(200).json({ ok: true, model: MODEL, fallbacks: MODELS.slice(1), usage: 'POST {"question": "..."}' });
  if (req.method !== 'POST') return res.status(405).json({ error: 'Use POST.' });

  const origin = req.headers.origin;
  if (!origin || !ALLOWED_ORIGINS.includes(origin)) return res.status(403).json({ error: 'Origin not allowed.' });

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'Server is missing GEMINI_API_KEY.' });

  let question = '';
  try {
    question = String((await readJson(req))?.question ?? '').trim();
  } catch {
    return res.status(400).json({ error: 'Send JSON: {"question": "..."}' });
  }
  if (!question) return res.status(400).json({ error: 'Empty question.' });
  if (question.length > MAX_QUESTION_CHARS) return res.status(400).json({ error: `Please keep questions under ${MAX_QUESTION_CHARS} characters.` });

  const ip = String(req.headers['x-forwarded-for'] ?? 'unknown').split(',')[0].trim();
  const evalToken = process.env.EVAL_TOKEN; // lets the offline evaluation run all questions without hitting the limit
  const isEval = Boolean(evalToken) && req.headers['x-eval-token'] === evalToken;
  if (!isEval && rateLimited(ip)) return res.status(429).json({ error: 'Rate limit reached for this free demo. Please try again in a few minutes.' });

  try {
    return res.status(200).json(await ask(question, apiKey));
  } catch (e) {
    if (e instanceof UpstreamError) {
      const busy = e.status === 503 || e.status === 429 || e.status === 504;
      return res.status(busy ? 503 : 502).json({ error: busy ? 'All free models are busy right now. Please try again in a minute — the recorded runs below show real answers.' : e.message });
    }
    if (e?.name === 'TimeoutError') return res.status(504).json({ error: 'The model took too long to answer. Please try again.' });
    console.error(e);
    return res.status(500).json({ error: 'Unexpected server error.' });
  }
}
