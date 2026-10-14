// Local stand-in for Vercel: serves api/ask.js with the same (req, res) helpers.
//   GEMINI_API_KEY=... node dev-server.mjs   →  http://localhost:3001/api/ask
import { createServer } from 'node:http';
import handler from './api/ask.js';

const PORT = Number(process.env.PORT ?? 3001);

createServer(async (req, res) => {
  if (!req.url.startsWith('/api/ask')) { res.writeHead(404).end('not found'); return; }
  res.status = (code) => { res.statusCode = code; return res; };
  res.json = (obj) => { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify(obj)); return res; };
  try {
    await handler(req, res);
  } catch (e) {
    console.error(e);
    res.statusCode = 500;
    res.end('crash');
  }
}).listen(PORT, () => console.log(`assistant API on http://localhost:${PORT}/api/ask`));
