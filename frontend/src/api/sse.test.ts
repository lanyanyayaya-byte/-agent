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

test('parses incremental MCP tool event metadata while preserving native tool events', () => {
  const base = { version: 1, session_id: 's1', run_id: 'r1', ts: '2026-11-25T08:00:00Z' };

  const nativeStart = parseRunEvent({
    ...base,
    type: 'tool_started',
    data: { tool_call_id: 'c-native', tool_name: 'read_text' },
  });
  expect(nativeStart).toEqual({
    session_id: 's1',
    run_id: 'r1',
    type: 'tool_started',
    data: { tool_call_id: 'c-native', tool_name: 'read_text' },
  });

  const mcpStart = parseRunEvent({
    ...base,
    type: 'tool_started',
    data: {
      tool_call_id: 'c-mcp',
      tool_name: 'mcp__srv_1__echo_text',
      source: 'mcp',
      server_id: 'srv-1',
      server_name: 'Local Stdio',
      original_tool_name: 'echo_text',
    },
  });
  expect(mcpStart).toEqual({
    session_id: 's1',
    run_id: 'r1',
    type: 'tool_started',
    data: {
      tool_call_id: 'c-mcp',
      tool_name: 'mcp__srv_1__echo_text',
      source: 'mcp',
      server_id: 'srv-1',
      server_name: 'Local Stdio',
      original_tool_name: 'echo_text',
    },
  });

  const mcpDone = parseRunEvent({
    ...base,
    type: 'tool_completed',
    data: {
      tool_call_id: 'c-mcp',
      summary: 'MCP 工具返回 128 字符。',
      source: 'mcp',
      server_id: 'srv-1',
      server_name: 'Local Stdio',
      original_tool_name: 'echo_text',
      duration_ms: 842,
    },
  });
  expect(mcpDone).toEqual({
    session_id: 's1',
    run_id: 'r1',
    type: 'tool_completed',
    data: {
      tool_call_id: 'c-mcp',
      summary: 'MCP 工具返回 128 字符。',
      source: 'mcp',
      server_id: 'srv-1',
      server_name: 'Local Stdio',
      original_tool_name: 'echo_text',
      duration_ms: 842,
    },
  });

  const mcpFail = parseRunEvent({
    ...base,
    type: 'tool_failed',
    data: {
      tool_call_id: 'c-mcp-2',
      code: 'MCP_TOOL_TIMEOUT',
      message: 'MCP 工具调用超时。',
      source: 'mcp',
      server_id: 'srv-1',
      server_name: 'Local Stdio',
      original_tool_name: 'slow_tool',
      duration_ms: 60012,
    },
  });
  expect(mcpFail).toEqual({
    session_id: 's1',
    run_id: 'r1',
    type: 'tool_failed',
    data: {
      tool_call_id: 'c-mcp-2',
      code: 'MCP_TOOL_TIMEOUT',
      message: 'MCP 工具调用超时。',
      source: 'mcp',
      server_id: 'srv-1',
      server_name: 'Local Stdio',
      original_tool_name: 'slow_tool',
      duration_ms: 60012,
    },
  });
});
