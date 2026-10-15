// Parity tests for the serverless tools (precomputed data) against values from the Python tools.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { runTool } from '../lib/tools.js';

const call = (name, args) => {
  const r = runTool(name, args);
  return { ...r, data: r.isError ? null : JSON.parse(r.output) };
};

test('cycle summary matches Python', () => {
  const { data } = call('get_cycle_summary', { profile_id: 74 });
  assert.equal(data.components.magnet.peak_measured_c, 91.9);
});

test('unknown cycle is a tool error, not a crash', () => {
  const r = call('get_cycle_summary', { profile_id: 999 });
  assert.equal(r.isError, true);
  assert.match(r.output, /does not exist/);
});

test('hot restarts', () => {
  const ids = call('find_hot_restarts', { min_delta_k: 10 }).data.cycles.map((c) => c.profile_id).sort();
  assert.deepEqual(ids, [47, 48, 53]);
});

test('what_if defaults omitted arguments like Python (regression for the 11/13 web eval bug)', () => {
  const r = call('what_if', { profile_id: 62, coolant_delta_c: 10 }).data;
  assert.equal(r.current_scale, 1);
  assert.equal(r.magnet.peak_change_k, 4.9);
  assert.equal(r.snapped, undefined);
});

test('what_if snaps to the precomputed grid and says so', () => {
  const r = call('what_if', { profile_id: 20, coolant_delta_c: 7, current_scale: 1.0 }).data;
  assert.equal(r.coolant_delta_c, 10);
  assert.ok(r.snapped.some((s) => s.includes('snapped')));
});

test('overheating threshold is rounded and reported', () => {
  const r = call('find_overheating', { threshold_c: 99.6, component: 'magnet', source: 'measured' }).data;
  assert.equal(r.threshold_c, 100);
  assert.ok(r.note);
});
