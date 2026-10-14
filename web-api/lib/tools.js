/** Assistant tools backed by precomputed data (exported by src/export_api_data.py).
 *  Outputs match the Python tools in src/tools.py; inputs that fall between precomputed
 *  values are snapped and the snap is reported back to the model. */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const DATA = join(dirname(fileURLToPath(import.meta.url)), '..', 'data');
const load = (f) => JSON.parse(readFileSync(join(DATA, f), 'utf8'));

const cycles = load('cycles.json');
const overheating = load('overheating.json');
const hotRestarts = load('hot_restarts.json');
const whatIf = load('what_if.json');
const modelCard = load('model_card.json');

const COMPONENTS = ['magnet', 'stator_yoke', 'stator_tooth', 'stator_winding'];
const validIds = Object.keys(cycles.summaries).map(Number).sort((a, b) => a - b);

export class ToolError extends Error {}

const nearest = (v, options) => options.reduce((b, o) => (Math.abs(o - v) < Math.abs(b - v) ? o : b), options[0]);

function cycleExists(id) {
  if (!cycles.summaries[String(id)]) {
    throw new ToolError(`Drive cycle ${id} does not exist. Valid ids: ${validIds[0]}–${validIds[validIds.length - 1]} (${validIds.length} cycles, with gaps).`);
  }
}

export const TOOLS = {
  list_drive_cycles: () => cycles.listing,

  get_cycle_summary: ({ profile_id }) => {
    cycleExists(profile_id);
    return cycles.summaries[String(profile_id)];
  },

  find_overheating: ({ threshold_c, component = 'magnet', source = 'virtual_sensor' }) => {
    if (!Number.isFinite(threshold_c)) throw new ToolError('threshold_c must be a number');
    if (!COMPONENTS.includes(component)) throw new ToolError(`component must be one of ${COMPONENTS.join(', ')}`);
    if (!['virtual_sensor', 'measured'].includes(source)) throw new ToolError("source must be 'virtual_sensor' or 'measured'");
    if (threshold_c < 40 || threshold_c > 160) throw new ToolError('threshold_c must be between 40 and 160 °C in the online version');
    const th = Math.round(threshold_c);
    const hits = overheating[`${component}|${source}|${th}`] ?? [];
    return {
      threshold_c: th, component, source, n_cycles_exceeding: hits.length, cycles: hits,
      ...(th !== threshold_c ? { note: `Threshold rounded to ${th} °C (online data is precomputed at 1 °C steps).` } : {}),
    };
  },

  find_hot_restarts: ({ min_delta_k = 10 }) => {
    const hits = hotRestarts.filter((c) => c.delta_k >= min_delta_k).sort((a, b) => b.delta_k - a.delta_k);
    return { min_delta_k, n_cycles: hits.length, cycles: hits };
  },

  // Defaults match src/tools.py; the model sometimes omits arguments it considers optional.
  what_if: ({ profile_id, coolant_delta_c = 0, current_scale = 1.0 }) => {
    if (!Number.isFinite(coolant_delta_c) || !Number.isFinite(current_scale)) throw new ToolError('coolant_delta_c and current_scale must be numbers');
    cycleExists(profile_id);
    if (!whatIf.profiles.includes(profile_id)) {
      throw new ToolError(`What-if is precomputed online only for the evaluation cycles ${whatIf.profiles.join(', ')}. Ask about one of those.`);
    }
    if (Math.abs(coolant_delta_c) > 30) throw new ToolError('coolant_delta_c must be within ±30 °C');
    if (current_scale < 0.5 || current_scale > 1.5) throw new ToolError('current_scale must be within 0.5–1.5');
    const dc = nearest(coolant_delta_c, whatIf.coolant);
    const cs = nearest(current_scale, whatIf.current);
    const r = whatIf.grid[`${profile_id}|${dc}|${cs.toFixed(1)}`];
    const notes = [];
    if (dc !== coolant_delta_c) notes.push(`coolant offset snapped to ${dc} °C (online grid: ${whatIf.coolant.join(', ')})`);
    if (cs !== current_scale) notes.push(`current scale snapped to ${cs} (online grid: ${whatIf.current.join(', ')})`);
    return notes.length ? { ...r, snapped: notes } : r;
  },

  model_card: () => modelCard,
};

/** Same schemas as the Python assistant (src/tools.py TOOL_SCHEMAS), in Gemini function format. */
export const TOOL_SCHEMAS = [
  { name: 'list_drive_cycles', description: 'List all recorded drive cycles (id, duration, max speed, start temperatures, peak magnet temperature measured and estimated, and whether the cycle is train/validation/test). Use it for overviews and to find cycles by property.', parameters: { type: 'object', properties: {}, required: [] } },
  { name: 'get_cycle_summary', description: 'Detailed summary of ONE drive cycle: operating conditions, start temperatures, and the virtual sensor\'s accuracy per component (peak measured vs estimated, MSE, max error). Use before making claims about a specific cycle.', parameters: { type: 'object', properties: { profile_id: { type: 'integer', description: 'Drive cycle id' } }, required: ['profile_id'] } },
  { name: 'find_overheating', description: "Find drive cycles and time windows where a component's temperature exceeds a threshold. source='virtual_sensor' uses the estimate a vehicle would have; source='measured' uses lab ground truth.", parameters: { type: 'object', properties: { threshold_c: { type: 'number', description: 'Temperature threshold in °C' }, component: { type: 'string', enum: COMPONENTS }, source: { type: 'string', enum: ['virtual_sensor', 'measured'] } }, required: ['threshold_c', 'component', 'source'] } },
  { name: 'find_hot_restarts', description: 'Find drive cycles that start hot: the magnet is at least min_delta_k hotter than the coolant at the first sample. Use for questions about hot restarts or start-up temperature differences instead of computing from list_drive_cycles.', parameters: { type: 'object', properties: { min_delta_k: { type: 'number', description: 'Minimum magnet-minus-coolant difference at start, in K' } }, required: ['min_delta_k'] } },
  { name: 'what_if', description: 'Re-run the virtual sensor on one drive cycle with coolant shifted by coolant_delta_c (°C, within ±30) and/or phase currents scaled by current_scale (0.5–1.5). Returns magnet and winding peak/mean before and after, plus extrapolation warnings that must be reported to the user.', parameters: { type: 'object', properties: { profile_id: { type: 'integer' }, coolant_delta_c: { type: 'number' }, current_scale: { type: 'number' } }, required: ['profile_id', 'coolant_delta_c', 'current_scale'] } },
  { name: 'model_card', description: 'How the virtual sensor works, its measured accuracy against published models, and known limitations. Use when asked how reliable or how built the estimates are.', parameters: { type: 'object', properties: {}, required: [] } },
].map((t) => ({ type: 'function', ...t }));

/** @returns {{output: string, isError: boolean}} */
export function runTool(name, args) {
  const fn = TOOLS[name];
  if (!fn) return { output: `Unknown tool ${name}`, isError: true };
  try {
    return { output: JSON.stringify(fn(args)), isError: false };
  } catch (e) {
    if (e instanceof ToolError) return { output: `Error: ${e.message}`, isError: true };
    throw e;
  }
}
