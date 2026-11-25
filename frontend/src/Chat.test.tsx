// @vitest-environment jsdom

import { invoke } from '@tauri-apps/api/core';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';
import { connectApiClient, type Message, type RunState, type RunSummary, type SessionSummary } from './api/apiClient';
import Chat from './Chat';

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn() }));

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  localStorage.clear();
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

const timestamp = '2026-11-22T08:30:00Z';

function run(sessionId: string, id: string, status: RunState, errorCode: string | null = null): RunSummary {
  return {
    run_id: id, session_id: sessionId, user_message_id: `user-${id}`, status, started_at: timestamp,
    finished_at: status === 'running' ? null : timestamp, error_code: errorCode,
  };
}

function message(record: RunSummary, content: string, role: Message['role'] = 'user'): Message {
  return {
    run_id: record.run_id, session_id: record.session_id, message_id: role === 'user' ? record.user_message_id : `saved-${record.run_id}`,
    role, status: 'committed', content, created_at: timestamp,
  };
}

async function setup() {
  vi.mocked(invoke).mockResolvedValue({ base_url: 'http://127.0.0.1:8766', ui_token: 'ui-test' });
  const sessions: SessionSummary[] = [];
  const messages: Message[] = [];
  const runs: RunSummary[] = [];
  const overrides = new Map<string, () => Promise<Response> | Response>();
  let stream: ReadableStream<Uint8Array>;
  const fetchApi = vi.fn(async (url: string, options: RequestInit) => {
    expect((options.headers as Record<string, string>).Authorization).toBe('Bearer ui-test');
    const { pathname, searchParams } = new URL(url);
    const path = pathname.slice('/api/v1'.length);
    const override = overrides.get(path);
    if (override) return override();
    if (path === '/mcp/summary') {
      return Response.json({
        total_servers: 0,
        enabled_servers: 0,
        connected_servers: 0,
        error_servers: 0,
        effective_allowed_tools: 0,
        max_tools_per_run: 32,
        exceeds_tool_limit: false,
      });
    }
    if (path === '/sessions') {
      if (options.method === 'GET') return Response.json(sessions);
      const session = { session_id: String(sessions.length + 1), title: JSON.parse(String(options.body)).title };
      sessions.push({ ...session, updated_at: timestamp, active_run_id: null, active_run_status: null });
      return Response.json({ ...session, created_at: timestamp }, { status: 201 });
    }
    if (path.endsWith('/runs/stream')) {
      expect((options.headers as Record<string, string>).Accept).toBe('text/event-stream');
      expect(options.method).toBe('POST');
      return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
    }
    const history = path.match(/^\/sessions\/([^/]+)\/(messages|runs|grants)$/);
    if (history) {
      const [, sessionId, kind] = history;
      if (kind === 'messages') return Response.json(messages.filter((row) => row.session_id === sessionId));
      if (kind === 'grants') return Response.json([]);
      const items = runs.filter((row) => row.session_id === sessionId);
      return Response.json(searchParams.has('cursor')
        ? { items: items.slice(1), next_cursor: null }
        : { items: items.slice(0, 1), next_cursor: items.length > 1 ? 'next' : null });
    }
    const result = path === '/runs/lookup' ? runs[0] : runs.find((row) => `/runs/${row.run_id}` === path);
    if (!result) throw new Error(`Unexpected request ${options.method} ${path}`);
    return Response.json({ ...result, pending_approval: null });
  });
  vi.stubGlobal('fetch', fetchApi);
  const client = await connectApiClient();

  async function create() {
    const title = `会话 ${sessions.length + 1}`;
    const button = await screen.findByRole<HTMLButtonElement>('button', { name: '新建会话' });
    await waitFor(() => expect(button.disabled).toBe(false));
    fireEvent.click(button);
    await screen.findByRole('option', { name: title });
  }

  async function start(sessionId: string, runId: string, question: string) {
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    stream = new ReadableStream<Uint8Array>({ start(value) { controller = value; } });
    fireEvent.change(screen.getByLabelText('问题'), { target: { value: question } });
    const send = screen.getByRole<HTMLButtonElement>('button', { name: '发送' });
    await waitFor(() => expect(send.disabled).toBe(false));
    fireEvent.click(send);
    const [url, options] = fetchApi.mock.lastCall!;
    expect(url).toContain(`/sessions/${sessionId}/runs/stream`);
    const request = JSON.parse(String(options.body));
    expect(request.message).toBe(question);
    expect(request.client_request_id).toMatch(/^[0-9a-f-]{36}$/);
    expect(screen.getByText('发送中')).toBeTruthy();
    return {
      requestId: request.client_request_id as string,
      push(type: string, data: object = {}, session = sessionId, id = runId) {
        controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({
          version: 1, type, session_id: session, run_id: id, ts: timestamp, data,
        })}\n\n`));
      },
      close: () => controller.close(),
    };
  }

  function save(sessionId: string, id: string, question: string, status: RunState, reply?: string) {
    const summary = run(sessionId, id, status, status === 'failed' ? 'MODEL_TIMEOUT' : null);
    runs.unshift(summary);
    messages.push(message(summary, question));
    if (reply) messages.push(message(summary, reply, 'assistant'));
  }
  const calls = (method: string) => fetchApi.mock.calls.filter(([, request]) => request.method === method);
  return { client, create, start, save, calls, overrides, sessions, messages, runs };
}

function select(id: string) {
  fireEvent.change(screen.getByLabelText('当前会话'), { target: { value: id } });
}

test('routes SSE by session and run and displays saved answers from history', async () => {
  const { client, create, start, save, calls } = await setup();
  render(<Chat client={client} modelConfigured />);
  await create();
  const a = await start('1', 'run-a', '问题 A');
  a.push('run_started', { client_request_id: a.requestId });
  a.push('message_delta', { channel: 'text', delta: '草稿 A' });
  expect((await screen.findByLabelText('回答草稿')).textContent).toBe('草稿 A');
  expect(screen.queryByLabelText('已保存回答')).toBeNull();

  await create();
  const b = await start('2', 'run-b', '问题 B');
  expect(a.requestId).not.toBe(b.requestId);
  save('1', 'run-a', '问题 A', 'completed', '数据库回答 A');
  await act(async () => {
    a.push('run_completed', { message_id: 'saved-run-a' });
    b.push('run_started', { client_request_id: b.requestId });
    b.push('message_delta', { channel: 'text', delta: 'wrong session' }, '1');
    b.push('message_delta', { channel: 'text', delta: 'wrong run' }, '2', 'run-a');
    b.push('message_delta', { channel: 'text', delta: '草稿 B' });
  });
  expect((await screen.findByLabelText('回答草稿')).textContent).toBe('草稿 B');
  expect(screen.queryByText('数据库回答 A')).toBeNull();
  save('2', 'run-b', '问题 B', 'failed');
  b.push('error', { code: 'MODEL_TIMEOUT', message: '模型响应超时。' });
  expect(await screen.findByText('MODEL_TIMEOUT')).toBeTruthy();
  expect(screen.queryByText('草稿 B')).toBeNull();
  expect(screen.queryByLabelText('已保存回答')).toBeNull();

  select('1');
  const answer = await screen.findByLabelText('已保存回答');
  expect(answer.textContent).toBe('数据库回答 A');
  expect(answer.getAttribute('data-message-id')).toBe('saved-run-a');
  expect(screen.queryByText('问题 B')).toBeNull();
  expect(calls('POST')).toHaveLength(4);
});

test.each(['cancelled', 'completed'] as const)('queries after cancellation and renders SQLite terminal state %s', async (terminal) => {
  const { client, create, start, save, overrides, calls, runs, messages } = await setup();
  render(<Chat client={client} modelConfigured />);
  await create();
  const pending = await start('1', 'run-a', '先生成再停止');
  pending.push('run_started', { client_request_id: pending.requestId });
  pending.push('message_delta', { channel: 'text', delta: '部分草稿' });
  expect(await screen.findByLabelText('回答草稿')).toBeTruthy();
  save('1', 'run-a', '先生成再停止', 'running');
  const snapshot = Promise.withResolvers<Response>();
  overrides.set('/runs/run-a', () => snapshot.promise);
  overrides.set('/runs/run-a/cancel', () => terminal === 'cancelled'
    ? Response.json({ run_id: 'run-a', status: 'cancelling' }, { status: 202 })
    : Response.json({ run_id: 'run-a', status: terminal, already_finished: true }));
  if (terminal === 'completed') {
    runs[0] = run('1', 'run-a', terminal);
    messages.push(message(runs[0], '数据库正式回答', 'assistant'));
  }
  fireEvent.click(screen.getByRole('button', { name: '停止生成' }));
  expect(calls('POST').at(-1)?.[0]).toContain('/runs/run-a/cancel');

  if (terminal === 'cancelled') {
    expect(await screen.findByText('取消中')).toBeTruthy();
    expect(screen.queryByLabelText('回答草稿')).toBeNull();
    runs[0] = run('1', 'run-a', terminal);
    snapshot.resolve(Response.json({ ...runs[0], pending_approval: null }));
    expect(await screen.findByText('已取消')).toBeTruthy();
    expect(screen.queryByLabelText('已保存回答')).toBeNull();
  } else {
    expect(await screen.findByText('已完成')).toBeTruthy();
    expect(screen.queryByLabelText('回答草稿')).toBeNull();
  }
});

test('reconciles a lost run_started using lookup, then reads SQLite without replaying', async () => {
  const { client, create, start, save, overrides, calls } = await setup();
  render(<Chat client={client} modelConfigured />);
  await create();
  const pending = await start('1', 'original-run', '首事件丢失');
  const lookup = Promise.withResolvers<Response>();
  overrides.set('/runs/lookup', () => lookup.promise);
  pending.close();
  expect(await screen.findByText('运行状态未知')).toBeTruthy();
  expect(calls('GET').at(-1)?.[0]).toContain(`/runs/lookup?session_id=1&client_request_id=${pending.requestId}`);
  save('1', 'original-run', '首事件丢失', 'completed', '落盘回答');
  lookup.resolve(Response.json({ ...run('1', 'original-run', 'completed'), pending_approval: null }));
  expect((await screen.findByLabelText('已保存回答')).textContent).toBe('落盘回答');
  expect(calls('GET').some(([url]) => url.endsWith('/runs/original-run'))).toBe(true);
  expect(calls('POST')).toHaveLength(2);
});

test('reload discovers an active run, joins paged history and polls for a terminal SQLite snapshot', async () => {
  const { client, save, sessions, runs, messages, calls, overrides } = await setup();
  sessions.push({ session_id: '1', title: '已保存会话', updated_at: timestamp, active_run_id: 'live', active_run_status: 'running' });
  save('1', 'old', '旧问题', 'cancelled');
  save('1', 'live', '仍在运行的问题', 'running');
  const current = Promise.withResolvers<Response>();
  overrides.set('/runs/live', () => current.promise);

  render(<Chat client={client} modelConfigured />);
  expect(await screen.findByText('已取消')).toBeTruthy();
  expect(screen.getByText('仍在运行的问题')).toBeTruthy();
  expect(screen.getByText('生成中')).toBeTruthy();
  expect(screen.queryByLabelText('回答草稿')).toBeNull();
  expect(screen.queryByLabelText('已保存回答')).toBeNull();
  expect(calls('GET').some(([url]) => url.includes('cursor=next'))).toBe(true);
  expect(calls('GET').some(([url]) => url.endsWith('/runs/live'))).toBe(true);
  expect(calls('GET').some(([url]) => url.endsWith('/runs/old'))).toBe(false);

  runs[0] = run('1', 'live', 'completed');
  messages.push(message(runs[0], '刷新后从 SQLite 读取的回答', 'assistant'));
  current.resolve(Response.json({ ...runs[0], pending_approval: null }));
  expect((await screen.findByLabelText('已保存回答')).textContent).toBe('刷新后从 SQLite 读取的回答');
  expect(calls('POST')).toHaveLength(0);
});

test('uses typed native grant commands and isolates list responses after switching sessions', async () => {
  const { client, create, overrides } = await setup();
  render(<Chat client={client} modelConfigured />);
  await create();
  const grant = { session_id: '1', grant_id: 'grant-a', scope: 'file', display_name: 'notes.md', operations: ['read'] };
  let items = [grant];
  overrides.set('/sessions/1/grants', () => Response.json(items));
  vi.mocked(invoke).mockResolvedValueOnce(grant);
  fireEvent.click(screen.getByRole('button', { name: '授权 TXT / MD' }));
  expect(invoke).toHaveBeenLastCalledWith('select_file_grant', { request: { session_id: '1', scope: 'file' } });
  expect(await screen.findByText('notes.md')).toBeTruthy();

  await create();
  expect(await screen.findByText('此会话暂无文件授权。')).toBeTruthy();
  expect(screen.queryByText('notes.md')).toBeNull();
  const picker = Promise.withResolvers<null>();
  vi.mocked(invoke).mockReturnValueOnce(picker.promise);
  fireEvent.click(screen.getByRole('button', { name: '授权文件夹' }));
  expect(invoke).toHaveBeenLastCalledWith('select_file_grant', { request: { session_id: '2', scope: 'directory' } });
  await act(() => { picker.resolve(null); });
  expect(screen.getByText('此会话暂无文件授权。')).toBeTruthy();

  select('1');
  const revoke = await screen.findByRole('button', { name: '撤销 notes.md' });
  vi.mocked(invoke).mockImplementationOnce(async () => { items = []; });
  fireEvent.click(revoke);
  expect(invoke).toHaveBeenLastCalledWith('revoke_file_grant', { request: { session_id: '1', grant_id: 'grant-a' } });
  expect(await screen.findByText('此会话暂无文件授权。')).toBeTruthy();
  expect(screen.queryByText('notes.md')).toBeNull();
});

test('displays safe tool progress and isolates tool events by both session and run', async () => {
  const { client, create, start, save } = await setup();
  render(<Chat client={client} modelConfigured />);
  await create();
  const a = await start('1', 'run-a', '发现并读取文件');
  await act(async () => {
    a.push('run_started', { client_request_id: a.requestId });
    a.push('tool_started', { tool_call_id: 'call-1', tool_name: 'list_files' });
  });
  const activity = () => screen.getByRole('list', { name: '工具执行' }).textContent;
  expect(activity()).toContain('list_files · 执行中');

  await create();
  const b = await start('2', 'run-b', '读取另一份文件');
  await act(async () => {
    b.push('run_started', { client_request_id: b.requestId });
    a.push('tool_completed', { tool_call_id: 'call-1', summary: '列出 1 项。', content: 'secret body' });
    b.push('tool_started', { tool_call_id: 'wrong', tool_name: 'foreign tool' }, '1');
    b.push('tool_started', { tool_call_id: 'wrong', tool_name: 'foreign tool' }, '2', 'run-a');
    b.push('tool_started', { tool_call_id: 'call-1', tool_name: 'read_text' });
    b.push('tool_failed', { tool_call_id: 'call-1', code: 'FILE_PERMISSION_DENIED', message: '文件未授权。' });
  });
  expect(activity()).toBe('read_text · 执行失败 · 文件未授权。');
  expect(screen.queryByText('foreign tool')).toBeNull();
  expect(screen.queryByText('secret body')).toBeNull();

  select('1');
  expect(activity()).toBe('list_files · 已完成 · 列出 1 项。');
  await act(async () => {
    a.push('tool_started', { tool_call_id: 'call-2', tool_name: 'read_text' });
    a.push('tool_completed', { tool_call_id: 'call-2', summary: '读取了 2 行。', content: 'secret body' });
    a.push('message_delta', { channel: 'text', delta: '模型回答' });
  });
  expect((await screen.findByLabelText('回答草稿')).textContent).toBe('模型回答');
  expect(activity()).toContain('read_text · 已完成 · 读取了 2 行。');
  save('1', 'run-a', '发现并读取文件', 'completed', '正式回答');
  a.push('run_completed', { message_id: 'saved-run-a' });
  expect((await screen.findByLabelText('已保存回答')).textContent).toBe('正式回答');
  expect(activity()).toContain('读取了 2 行。');

  select('2');
  expect(activity()).toBe('read_text · 执行失败 · 文件未授权。');
  save('2', 'run-b', '读取另一份文件', 'failed');
  b.push('error', { code: 'FILE_PERMISSION_DENIED', message: '文件未授权。' });
  await screen.findByText('失败', { exact: true });
  expect(screen.queryByLabelText('已保存回答')).toBeNull();
});

test('supports RAG textbook Q&A mode, renders structured citations and persisted tool calls', async () => {
  const { client, create, start, overrides, runs, messages } = await setup();
  const docId = 'a'.repeat(32);
  overrides.set('/knowledge/documents', () => Response.json([
    {
      document_id: docId,
      name: '数理统计讲义.pdf',
      format: 'pdf',
      management_status: 'imported',
      created_at: timestamp,
      updated_at: timestamp,
      version_count: 1,
      latest_size_bytes: 1024,
    },
  ]));

  render(<Chat client={client} modelConfigured />);
  await create();

  // Enable RAG textbook mode
  const toggle = screen.getByLabelText('结合教材知识库');
  fireEvent.click(toggle);
  expect(await screen.findByText('数理统计讲义.pdf')).toBeTruthy();
  fireEvent.change(screen.getByLabelText('教材问答召回策略'), { target: { value: 'per_document' } });

  const turn = await start('1', 'run-rag', '对比最大似然估计与贝叶斯估计');
  const citationItem = {
    evidence_id: `ev1:${docId}:1:chunk:${'1'.repeat(64)}:${'2'.repeat(64)}`,
    chunk_id: `chunk:${'1'.repeat(64)}`,
    chunk_set_id: 1,
    chunk_index: 0,
    document_id: docId,
    document_version_id: 'b'.repeat(32),
    textbook_name: '数理统计讲义.pdf',
    format: 'pdf' as const,
    section_path: ['第三章 参数估计'],
    location_summary: '第 46 页',
    location_precision: 'block',
    location_status: 'available',
    warnings: [{ code: 'PDF_UNMAPPED_GLYPHS', message: 'Unmapped math glyphs', physical_page: 46 }],
    snippet: '最大似然估计通过最大化 $L(\\theta)$ 求解。',
    status: 'resolved',
    historical: false,
    parse_spec_current: true,
  };

  await act(async () => {
    turn.push('run_started', { client_request_id: turn.requestId });
    turn.push('tool_started', { tool_call_id: 'rag-call-1', tool_name: 'search_knowledge' });
    turn.push('tool_completed', { tool_call_id: 'rag-call-1', summary: '混合检索 · 命中 1 条证据。' });
    turn.push('run_citations', {
      citations: [citationItem],
      retrieval_query: '对比最大似然估计与贝叶斯估计',
      coverage_mode: 'per_document',
      channel_status: 'hybrid',
    });
    turn.push('message_delta', { channel: 'text', delta: '根据教材 [引用 1]，似然函数为 $L(\\theta)$。' });
  });

  expect(await screen.findByText('第 46 页')).toBeTruthy();
  expect(screen.getByText('已核验原件')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '查看证据原文' }));
  expect(screen.getByText(/最大似然估计通过最大化/)).toBeTruthy();

  // Persist run with tool_calls and citations in SQLite mock
  const completedSummary: RunSummary = {
    ...run('1', 'run-rag', 'completed'),
    tool_calls: [{
      tool_call_id: 'rag-call-1',
      tool_name: 'search_knowledge',
      status: 'completed',
      summary: '混合检索 · 命中 1 条证据。',
      error_code: null,
    }],
    citations: [{ ...citationItem, status: 'source_removed' }],
  };
  runs.unshift(completedSummary);
  messages.push(message(completedSummary, '对比最大似然估计与贝叶斯估计'));
  messages.push(message(completedSummary, '根据教材 [引用 1]，似然函数为 $L(\\theta)$。', 'assistant'));

  await act(async () => {
    turn.push('run_completed', { message_id: 'saved-run-rag' });
  });
  expect(await screen.findByLabelText('已保存回答')).toBeTruthy();
});

test('sends explicit mcp options, keeps RAG and MCP switches independent, and restores MCP tool cards from SSE and history', async () => {
  const { client, create, overrides, calls, runs, messages } = await setup();
  overrides.set('/mcp/summary', () => Response.json({
    total_servers: 1,
    enabled_servers: 1,
    connected_servers: 1,
    error_servers: 0,
    effective_allowed_tools: 3,
    max_tools_per_run: 32,
    exceeds_tool_limit: false,
  }));

  render(<Chat client={client} modelConfigured />);
  await create();

  // MCP defaults to OFF and displays real available tool count from summary
  const mcpToggle = screen.getByLabelText<HTMLInputElement>('启用 MCP 工具');
  const ragToggle = screen.getByLabelText<HTMLInputElement>('结合教材知识库');
  expect(mcpToggle.checked).toBe(false);
  expect(ragToggle.checked).toBe(false);
  expect(await screen.findByText('MCP 工具（3 个可用）')).toBeTruthy();

  // Sending with MCP OFF explicitly sends { mcp: { enabled: false, allow_external_data_egress: false } }
  let streamCtrl!: ReadableStreamDefaultController<Uint8Array>;
  overrides.set('/sessions/1/runs/stream', () => {
    const stream = new ReadableStream<Uint8Array>({ start(c) { streamCtrl = c; } });
    return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
  });
  fireEvent.change(screen.getByLabelText('问题'), { target: { value: '普通聊天消息' } });
  fireEvent.click(screen.getByRole('button', { name: '发送' }));
  await waitFor(() => expect(calls('POST').some(([u]) => u.includes('/runs/stream'))).toBe(true));
  const offBody = JSON.parse(String(calls('POST').at(-1)?.[1].body));
  expect(offBody.mcp).toEqual({ enabled: false, allow_external_data_egress: false });
  expect(offBody.knowledge).toBeUndefined();

  const pushOff = (type: string, data: object) => streamCtrl.enqueue(new TextEncoder().encode(
    `data: ${JSON.stringify({ version: 1, type, session_id: '1', run_id: 'r-off', ts: timestamp, data })}\n\n`,
  ));
  await act(async () => {
    pushOff('run_started', { client_request_id: offBody.client_request_id });
    const doneRun = run('1', 'r-off', 'completed');
    runs.unshift(doneRun);
    messages.push(message(doneRun, '普通聊天消息'));
    messages.push(message(doneRun, '普通回复', 'assistant'));
    pushOff('run_completed', { message_id: 'saved-r-off' });
  });
  expect(await screen.findByText('普通回复')).toBeTruthy();

  // Toggle MCP ON independently (RAG stays OFF) -> remembered in localStorage
  fireEvent.click(mcpToggle);
  expect(mcpToggle.checked).toBe(true);
  expect(ragToggle.checked).toBe(false);
  expect(localStorage.getItem('knowledgeagent.mcp_enabled')).toBe('true');

  fireEvent.change(screen.getByLabelText('问题'), { target: { value: '调用外部 MCP 工具' } });
  fireEvent.click(screen.getByRole('button', { name: '发送' }));
  await waitFor(() => {
    const postStreams = calls('POST').filter(([u]) => u.includes('/runs/stream'));
    expect(postStreams).toHaveLength(2);
  });
  const onBody = JSON.parse(String(calls('POST').filter(([u]) => u.includes('/runs/stream')).at(-1)?.[1].body));
  expect(onBody.mcp).toEqual({ enabled: true, allow_external_data_egress: false });

  // Stream SSE MCP tool_started + tool_completed with source/server_name/original_tool_name/duration_ms
  const pushOn = (type: string, data: object) => streamCtrl.enqueue(new TextEncoder().encode(
    `data: ${JSON.stringify({ version: 1, type, session_id: '1', run_id: 'r-mcp', ts: timestamp, data })}\n\n`,
  ));
  await act(async () => {
    pushOn('run_started', { client_request_id: onBody.client_request_id });
    pushOn('tool_started', {
      tool_call_id: 'mcp-call-1',
      tool_name: 'mcp__srv_1__echo_text',
      source: 'mcp',
      server_id: 'srv-1',
      server_name: 'Local Test MCP',
      original_tool_name: 'echo_text',
    });
  });
  const liveCard = screen.getByRole('list', { name: '工具执行' }).textContent ?? '';
  expect(liveCard).toContain('MCP');
  expect(liveCard).toContain('Local Test MCP');
  expect(liveCard).toContain('echo_text');
  expect(liveCard).toContain('执行中');

  // Toggling MCP OFF mid-run does not break the active run
  fireEvent.click(mcpToggle);
  expect(mcpToggle.checked).toBe(false);

  const savedMcpRun: RunSummary = {
    ...run('1', 'r-mcp', 'completed'),
    tool_calls: [{
      tool_call_id: 'mcp-call-1',
      tool_name: 'mcp__srv_1__echo_text',
      status: 'completed',
      summary: 'MCP 工具返回 42 字符。',
      error_code: null,
      source: 'mcp',
      server_id: 'srv-1',
      server_name: 'Local Test MCP',
      original_tool_name: 'echo_text',
      duration_ms: 840,
    }],
  };
  runs.unshift(savedMcpRun);
  messages.push(message(savedMcpRun, '调用外部 MCP 工具'));
  messages.push(message(savedMcpRun, 'MCP 工具回答完成。', 'assistant'));

  await act(async () => {
    pushOn('tool_completed', {
      tool_call_id: 'mcp-call-1',
      summary: 'MCP 工具返回 42 字符。',
      source: 'mcp',
      server_id: 'srv-1',
      server_name: 'Local Test MCP',
      original_tool_name: 'echo_text',
      duration_ms: 840,
    });
    pushOn('run_completed', { message_id: 'saved-r-mcp' });
  });

  expect(await screen.findByText('MCP 工具回答完成。')).toBeTruthy();

  // Reopen / remount Chat to verify historical MCP tool call card restoration from SQLite runs
  cleanup();
  render(<Chat client={client} modelConfigured />);
  expect(await screen.findByText('MCP 工具回答完成。')).toBeTruthy();
  const restored = screen.getByRole('list', { name: '工具执行' }).textContent ?? '';
  expect(restored).toContain('MCP');
  expect(restored).toContain('Local Test MCP');
  expect(restored).toContain('echo_text');
  expect(restored).toContain('已完成');
  expect(restored).toContain('840ms');
  expect(restored).toContain('MCP 工具返回 42 字符。');
});

test('requires non-persistent one-turn egress confirmation for RAG+MCP, historical protected turns, and backend 403 without auto-retry', async () => {
  const { client, create, overrides, calls, runs, messages } = await setup();
  overrides.set('/mcp/summary', () => Response.json({
    total_servers: 1,
    enabled_servers: 1,
    connected_servers: 1,
    error_servers: 0,
    effective_allowed_tools: 2,
    max_tools_per_run: 32,
    exceeds_tool_limit: false,
  }));
  overrides.set('/knowledge/documents', () => Response.json([]));

  let streamCtrl!: ReadableStreamDefaultController<Uint8Array>;
  overrides.set('/sessions/1/runs/stream', () => {
    const stream = new ReadableStream<Uint8Array>({ start(c) { streamCtrl = c; } });
    return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
  });

  render(<Chat client={client} modelConfigured />);
  await create();

  // Turn ON both RAG and MCP
  fireEvent.click(screen.getByLabelText('结合教材知识库'));
  fireEvent.click(screen.getByLabelText('启用 MCP 工具'));

  // Attempt send -> MUST open confirmation dialog instead of calling backend!
  fireEvent.change(screen.getByLabelText('问题'), { target: { value: '结合教材与外部工具分析' } });
  fireEvent.click(screen.getByRole('button', { name: '发送' }));

  const dialog = await screen.findByRole('dialog', { name: '本轮外部数据外发确认' });
  expect(dialog.textContent).toContain('当前对话可能包含本地文件或教材内容');
  expect(screen.getByRole('button', { name: '关闭教材 RAG' })).toBeTruthy();
  expect(calls('POST').filter(([u]) => u.includes('/runs/stream'))).toHaveLength(0);

  // Confirm one-turn egress -> sends allow_external_data_egress: true ONCE, never persists to localStorage
  fireEvent.click(screen.getByRole('button', { name: '仅本轮授权必要的数据外发' }));
  await waitFor(() => expect(calls('POST').filter(([u]) => u.includes('/runs/stream'))).toHaveLength(1));
  const firstBody = JSON.parse(String(calls('POST').filter(([u]) => u.includes('/runs/stream'))[0][1].body));
  expect(firstBody.mcp).toEqual({ enabled: true, allow_external_data_egress: true });
  expect(JSON.stringify(localStorage)).not.toContain('allow_external_data_egress');

  // Complete turn 1 with a historical search_knowledge tool call
  const ragHistoryRun: RunSummary = {
    ...run('1', 'r-rag-1', 'completed'),
    tool_calls: [{
      tool_call_id: 'rag-tc-1',
      tool_name: 'search_knowledge',
      status: 'completed',
      summary: '命中 1 条教材证据。',
      error_code: null,
      source: 'native',
    }],
  };
  runs.unshift(ragHistoryRun);
  messages.push(message(ragHistoryRun, '结合教材与外部工具分析'));
  messages.push(message(ragHistoryRun, '第一轮结合教材完成。', 'assistant'));
  await act(async () => {
    const push = (type: string, data: object) => streamCtrl.enqueue(new TextEncoder().encode(
      `data: ${JSON.stringify({ version: 1, type, session_id: '1', run_id: 'r-rag-1', ts: timestamp, data })}\n\n`,
    ));
    push('run_started', { client_request_id: firstBody.client_request_id });
    push('run_completed', { message_id: 'saved-r-rag-1' });
  });
  expect(await screen.findByText('第一轮结合教材完成。')).toBeTruthy();

  // Turn OFF RAG on Turn 2 -> cross-turn historical RAG context STILL requires confirmation,
  // and MUST NOT offer "关闭教材 RAG" as a fix because history is already tainted!
  fireEvent.click(screen.getByLabelText('结合教材知识库'));
  fireEvent.change(screen.getByLabelText('问题'), { target: { value: '第二轮继续问' } });
  fireEvent.click(screen.getByRole('button', { name: '发送' }));

  const histDialog = await screen.findByRole('dialog', { name: '本轮外部数据外发确认' });
  expect(histDialog.textContent).toContain('会话历史包含教材检索上下文');
  expect(screen.queryByRole('button', { name: '关闭教材 RAG' })).toBeNull();
  expect(calls('POST').filter(([u]) => u.includes('/runs/stream'))).toHaveLength(1);

  // Start a new clean session via the dialog button -> preserves draft input!
  fireEvent.click(screen.getByRole('button', { name: '开启新对话' }));
  await waitFor(() => expect(screen.getByLabelText<HTMLTextAreaElement>('问题').value).toBe('第二轮继续问'));

  // Simulate backend returning 403 MCP_EGRESS_AUTHORIZATION_REQUIRED (e.g., legacy unverifiable session)
  overrides.set('/sessions/2/runs/stream', () => Response.json(
    { error: { code: 'MCP_EGRESS_AUTHORIZATION_REQUIRED', message: '旧版本历史上下文需要授权外发。' } },
    { status: 403 },
  ));
  const sendBtn = screen.getByRole<HTMLButtonElement>('button', { name: '发送' });
  await waitFor(() => expect(sendBtn.disabled).toBe(false));
  fireEvent.click(sendBtn);

  // Must preserve user input, show dialog, and NEVER auto-retry!
  const backendDialog = await screen.findByRole('dialog', { name: '本轮外部数据外发确认' });
  expect(backendDialog.textContent).toContain('旧版本历史上下文需要授权外发。');
  expect(screen.getByLabelText<HTMLTextAreaElement>('问题').value).toBe('第二轮继续问');
  expect(calls('POST').filter(([u]) => u.includes('/sessions/2/runs/stream'))).toHaveLength(1);
});

test('shows tool limit warning, distinct Chinese MCP error messages with jump link, and keeps plain chat working when MCP summary fails', async () => {
  const { client, create, overrides, runs, messages } = await setup();
  const onOpenMcp = vi.fn();

  // 1) Over-limit summary warning + jump button
  overrides.set('/mcp/summary', () => Response.json({
    total_servers: 2,
    enabled_servers: 2,
    connected_servers: 2,
    error_servers: 0,
    effective_allowed_tools: 35,
    max_tools_per_run: 32,
    exceeds_tool_limit: true,
  }));

  render(<Chat client={client} modelConfigured onOpenMcpPanel={onOpenMcp} />);
  await create();
  fireEvent.click(screen.getByLabelText('启用 MCP 工具'));
  expect(await screen.findByText(/超过单轮上限（32）/)).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '管理 MCP 工具' }));
  expect(onOpenMcp).toHaveBeenCalledTimes(1);

  // 2) Distinct Chinese MCP error banner in history (never shown as generic DeepSeek failure)
  const failedMcpRun = run('1', 'r-timeout', 'failed', 'MCP_TOOL_TIMEOUT');
  runs.unshift(failedMcpRun);
  messages.push(message(failedMcpRun, '触发超时'));
  cleanup();

  // 3) Even when /mcp/summary returns HTTP 500, ordinary chat still loads and displays friendly MCP error
  overrides.set('/mcp/summary', () => new Response('boom', { status: 500 }));
  render(<Chat client={client} modelConfigured onOpenMcpPanel={onOpenMcp} />);
  expect(await screen.findByText(/MCP 工具调用超时，外部执行结果未知/)).toBeTruthy();
  expect(screen.queryByText('DeepSeek 调用失败。')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: '前往 MCP 服务' }));
  expect(onOpenMcp).toHaveBeenCalledTimes(2);
});
