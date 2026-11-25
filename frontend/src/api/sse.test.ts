import { expect, test, vi } from 'vitest';
import { parseRunEvent } from './chatEvents';
import { readSse } from './sse';

test('reassembles UTF-8, CRLF, comments and consecutive JSON frames across arbitrary byte boundaries', async () => {
  const base = { version: 1, session_id: 'session', run_id: 'run', ts: '2026-11-22T08:30:00Z' };
  const started = { ...base, type: 'run_started', data: { client_request_id: 'request' } };
  const delta = { ...base, type: 'message_delta', data: { channel: 'text', delta: '你好🌍' } };
  const reasoning = { ...delta, data: { channel: 'reasoning', delta: 'hidden' } };
  const invalid = 'data: {broken}\n\n';
  const unknown = { ...base, type: 'future_type', data: {} };
  const completed = { ...base, type: 'run_completed', data: { message_id: 'message' } };
  const frame = (value: unknown, newline = '\r\n') => `data: ${JSON.stringify(value)}${newline}${newline}`;

  const encoder = new TextEncoder();
  const fragmented = encoder.encode(frame(started) + ': keepalive\r\n\r\n' + frame(delta));
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const byte of fragmented) controller.enqueue(Uint8Array.of(byte));
      controller.enqueue(encoder.encode(invalid + frame(reasoning, '\n') + frame(unknown) + frame(completed)));
      controller.enqueue(encoder.encode('data: {"truncated":true}'));
      controller.close();
    },
  });

  const debug = vi.spyOn(console, 'debug').mockImplementation(() => undefined);
  try {
    const events = [];
    for await (const value of readSse(body)) {
      const event = parseRunEvent(value);
      if (event) events.push(event);
    }
    expect(events.map((event) => [event.type, event.data])).toEqual([
      ['run_started', { client_request_id: 'request' }],
      ['message_delta', { delta: '你好🌍' }],
      ['run_completed', { message_id: 'message' }],
    ]);
    expect(debug.mock.calls).toEqual([
      ['Ignored malformed SSE data.'],
      ['Ignored an unsupported SSE event type.'],
    ]);
  } finally {
    debug.mockRestore();
  }
});
