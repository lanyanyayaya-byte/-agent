// @vitest-environment jsdom

import { invoke, isTauri } from '@tauri-apps/api/core';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import App from './App';

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn(), isTauri: vi.fn() }));

beforeEach(() => {
  vi.mocked(isTauri).mockReturnValue(true);
});

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  localStorage.clear();
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

const emptyMcpSummary = {
  total_servers: 0,
  enabled_servers: 0,
  connected_servers: 0,
  error_servers: 0,
  effective_allowed_tools: 0,
  max_tools_per_run: 32,
  exceeds_tool_limit: false,
};

test('keeps the active session and SSE draft while the library is open', async () => {
  vi.mocked(invoke).mockResolvedValue({ base_url: 'http://127.0.0.1:8766', ui_token: 'ui-test' });
  const sessionId = 'session-a';
  const runId = 'run-a';
  const ts = '2026-11-22T08:30:00Z';
  let streamController!: ReadableStreamDefaultController<Uint8Array>;
  let requestId = '';
  let streamSignal: AbortSignal | null | undefined;
  const stream = new ReadableStream<Uint8Array>({ start(controller) { streamController = controller; } });
  vi.stubGlobal('fetch', (url: string, options: RequestInit) => {
    const path = new URL(url).pathname.replace('/api/v1', '');
    if (path === '/status') return Response.json({ status: 'ready', database: 'ready', model_configured: true });
    if (path === '/sessions') return Response.json([{
      session_id: sessionId, title: '测试会话', updated_at: ts, active_run_id: null, active_run_status: null,
    }]);
    if (path === `/sessions/${sessionId}/runs`) return Response.json({ items: [], next_cursor: null });
    if (path === '/mcp/summary') return Response.json(emptyMcpSummary);
    if (path === '/mcp/servers') return Response.json([]);
    if (path === '/skills') return Response.json([]);
    if (path === `/sessions/${sessionId}/workspace`) return Response.json({ session_id: sessionId, files: [], artifacts: [] });
    if (path === '/knowledge/index-status') {
      return Response.json({
        total_documents: 0, ready_hybrid: 0, ready_keyword: 0, in_progress: 0,
        pending: 0, failed: 0, unsearchable: 0, missing_for_completion: 0,
        active_job_id: null, siliconflow_configured: false, documents: [],
        settings: {
          auto_index_on_import: true, cloud_embedding_enabled: false,
          cloud_rerank_enabled: true, enable_bilingual_rewrite: true,
          default_top_k: 8, compare_top_k: 12, rerank_candidates: 24, min_vector_similarity: 0.25,
        },
      });
    }
    if ([`/sessions/${sessionId}/messages`, `/sessions/${sessionId}/grants`, '/knowledge/documents'].includes(path)) {
      return Response.json([]);
    }
    if (path === `/sessions/${sessionId}/runs/stream`) {
      requestId = JSON.parse(String(options.body)).client_request_id;
      streamSignal = options.signal;
      return new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
    }
    throw new Error(`Unexpected request ${path}`);
  });
  function push(type: string, data: object) {
    const event = JSON.stringify({ version: 1, type, session_id: sessionId, run_id: runId, ts, data });
    streamController.enqueue(new TextEncoder().encode(`data: ${event}\n\n`));
  }

  render(<App />);
  fireEvent.change(await screen.findByLabelText('问题'), { target: { value: '测试问题' } });
  const send = screen.getByRole<HTMLButtonElement>('button', { name: '发送' });
  await waitFor(() => expect(send.disabled).toBe(false));
  fireEvent.click(send);
  push('run_started', { client_request_id: requestId });
  push('message_delta', { channel: 'text', delta: '第一段草稿' });
  await screen.findByText('第一段草稿');

  fireEvent.click(screen.getByRole('button', { name: '教材知识库' }));
  await screen.findByText(/还没有教材/);
  expect(screen.queryByRole('button', { name: '停止生成' })).toBeNull();
  expect(streamSignal?.aborted).toBe(false);
  push('message_delta', { channel: 'text', delta: '，第二段草稿' });

  fireEvent.click(screen.getByRole('button', { name: '聊天' }));
  await screen.findByText('第一段草稿，第二段草稿');
  expect(screen.getByRole<HTMLSelectElement>('combobox', { name: '当前会话' }).value).toBe(sessionId);
  expect(screen.getByRole('heading', { name: '文件授权' })).toBeTruthy();

  fireEvent.click(screen.getByRole('button', { name: 'MCP 服务' }));
  await screen.findByText('尚未配置任何 MCP 服务');
  expect(streamSignal?.aborted).toBe(false);
  fireEvent.click(screen.getByRole('button', { name: '聊天' }));
  expect(screen.getByText('第一段草稿，第二段草稿')).toBeTruthy();
  expect(streamSignal?.aborted).toBe(false);
  streamController.close();
});

test('renders the browser preview without contacting the desktop host or backend', () => {
  vi.mocked(isTauri).mockReturnValue(false);
  const fetchApi = vi.fn();
  vi.stubGlobal('fetch', fetchApi);

  render(<App />);

  expect(screen.getByRole('heading', { name: '网页预览' })).toBeTruthy();
  expect((screen.getByRole('button', { name: '新建会话' }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole('button', { name: '发送' }) as HTMLButtonElement).disabled).toBe(true);
  expect(screen.queryByRole('button', { name: '重试' })).toBeNull();
  expect(invoke).not.toHaveBeenCalled();
  expect(fetchApi).not.toHaveBeenCalled();
});

test('requests the host connection, authenticates the status call and retries after failure', async () => {
  const connection = { base_url: 'http://127.0.0.1:8766', ui_token: crypto.randomUUID() };
  const opening = Promise.withResolvers<typeof connection>();
  const firstRequest = Promise.withResolvers<Response>();
  vi.mocked(invoke).mockReturnValueOnce(opening.promise).mockResolvedValueOnce(connection);
  const fetchStatus = vi.fn()
    .mockReturnValueOnce(firstRequest.promise)
    .mockResolvedValueOnce(Response.json({ status: 'ready', database: 'not_configured', model_configured: false }));
  vi.stubGlobal('fetch', (url: string, options: RequestInit) => {
    if (url.endsWith('/sessions')) return Promise.resolve(Response.json([]));
    if (url.endsWith('/mcp/summary')) return Promise.resolve(Response.json(emptyMcpSummary));
    if (url.endsWith('/skills')) return Promise.resolve(Response.json([]));
    return fetchStatus(url, options);
  });

  render(<App />);
  expect(screen.getByText('starting_sidecar')).toBeTruthy();
  expect(invoke).toHaveBeenCalledWith('get_backend_connection');
  expect(fetchStatus).not.toHaveBeenCalled();

  opening.resolve(connection);
  expect(await screen.findByText('connecting_backend')).toBeTruthy();
  expect(fetchStatus).toHaveBeenCalledWith(`${connection.base_url}/api/v1/status`, expect.objectContaining({
    headers: { Accept: 'application/json', Authorization: `Bearer ${connection.ui_token}` },
    method: 'GET',
    redirect: 'error',
  }));

  firstRequest.reject(new TypeError('Failed to fetch'));
  expect(await screen.findByText('connection_error')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '重试' }));
  expect(await screen.findByText('ready')).toBeTruthy();
  expect(invoke).toHaveBeenCalledTimes(2);
  expect(fetchStatus).toHaveBeenCalledTimes(2);
});

test('shows an authentication rejection without reflecting backend error text', async () => {
  vi.mocked(invoke).mockResolvedValue({ base_url: 'http://127.0.0.1:8766', ui_token: crypto.randomUUID() });
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(Response.json({ error: { message: 'internal detail' } }, { status: 401 })));

  render(<App />);
  expect(await screen.findByText('后端拒绝了连接凭证（HTTP 401）。')).toBeTruthy();
  expect(screen.queryByText('internal detail')).toBeNull();
});

test('submits the password only to the credential IPC and clears it immediately', async () => {
  const key = crypto.randomUUID();
  const save = Promise.withResolvers<{ configured: boolean }>();
  vi.mocked(invoke)
    .mockResolvedValueOnce({ base_url: 'http://127.0.0.1:8766', ui_token: crypto.randomUUID() })
    .mockReturnValueOnce(save.promise);
  const fetchStatus = vi.fn().mockResolvedValue(Response.json({
    status: 'ready', database: 'not_configured', model_configured: false,
  }));
  vi.stubGlobal('fetch', (url: string, options: RequestInit) => {
    if (url.endsWith('/sessions')) return Promise.resolve(Response.json([]));
    if (url.endsWith('/mcp/summary')) return Promise.resolve(Response.json(emptyMcpSummary));
    if (url.endsWith('/skills')) return Promise.resolve(Response.json([]));
    return fetchStatus(url, options);
  });

  render(<App />);
  const field = await screen.findByLabelText('DeepSeek API Key') as HTMLInputElement;
  expect(field.type).toBe('password');
  fireEvent.change(field, { target: { value: key } });
  fireEvent.submit(field.form!);
  expect(invoke).toHaveBeenCalledWith('save_provider_credential', { provider: 'deepseek', apiKey: key });
  expect(field.value).toBe('');
  expect(fetchStatus).toHaveBeenCalledTimes(1);
  expect(JSON.stringify(fetchStatus.mock.calls)).not.toContain(key);
  expect(JSON.stringify(localStorage)).not.toContain(key);

  save.reject(new Error(key));
  expect(await screen.findByText('未能提交 DeepSeek 凭据，请检查输入后重试。')).toBeTruthy();
  expect(screen.queryByText(key)).toBeNull();
  expect(field.value).toBe('');
});

test('saves, tests, rotates and clears SiliconFlow without changing DeepSeek', async () => {
  const initialKey = crypto.randomUUID();
  const rotatedKey = crypto.randomUUID();
  const saved = Promise.withResolvers<{ configured: boolean }>();
  vi.mocked(invoke)
    .mockResolvedValueOnce({ base_url: 'http://127.0.0.1:8766', ui_token: crypto.randomUUID() })
    .mockReturnValueOnce(saved.promise)
    .mockResolvedValueOnce({ model: 'BAAI/bge-m3', dimensions: 1024 })
    .mockResolvedValueOnce({ configured: true })
    .mockRejectedValueOnce('EMBEDDING_AUTH_FAILED')
    .mockResolvedValueOnce({ configured: false });
  const fetchBackend = vi.fn((url: string, _options: RequestInit) =>
    Promise.resolve(Response.json(url.endsWith('/sessions') ? [] : {
      status: 'ready', database: 'ready', model_configured: true, siliconflow_configured: false,
    })));
  vi.stubGlobal('fetch', fetchBackend);

  render(<App />);
  const field = await screen.findByLabelText<HTMLInputElement>('SiliconFlow API Key');
  const testButton = screen.getByRole<HTMLButtonElement>('button', { name: '测试连接' });
  expect(field.type).toBe('password');
  expect(screen.getByText('SiliconFlow 未配置')).toBeTruthy();
  expect(testButton.disabled).toBe(true);

  fireEvent.change(field, { target: { value: initialKey } });
  fireEvent.click(screen.getByRole('button', { name: '保存 Key' }));
  expect(invoke).toHaveBeenLastCalledWith('save_provider_credential', { provider: 'siliconflow', apiKey: initialKey });
  expect(field.value).toBe('');
  saved.resolve({ configured: true });
  await screen.findByText('SiliconFlow 已配置');

  fireEvent.click(testButton);
  expect(invoke).toHaveBeenLastCalledWith('test_siliconflow_connection');
  await screen.findByText('连接成功，模型 BAAI/bge-m3，向量维度 1024。');

  fireEvent.change(field, { target: { value: rotatedKey } });
  fireEvent.click(screen.getByRole('button', { name: '更新 Key' }));
  expect(invoke).toHaveBeenLastCalledWith('save_provider_credential', { provider: 'siliconflow', apiKey: rotatedKey });
  await screen.findByText('已保存 SiliconFlow 凭据。');
  expect(screen.queryByText(/连接成功/)).toBeNull();
  fireEvent.click(testButton);
  await screen.findByText('SiliconFlow API Key 无效或已失效（401），请更新后重试。');

  fireEvent.click(screen.getByRole('button', { name: '清除 Key' }));
  expect(invoke).toHaveBeenLastCalledWith('delete_provider_credential', { provider: 'siliconflow' });
  await screen.findByText('SiliconFlow 未配置');
  expect(screen.getByText('已配置')).toBeTruthy();
  expect(screen.getByRole('button', { name: '删除凭据' })).toBeTruthy();
  const storage = JSON.stringify([localStorage, sessionStorage, fetchBackend.mock.calls]);
  for (const key of [initialKey, rotatedKey]) expect(storage).not.toContain(key);
});

test('reports an isolated SiliconFlow restore error with an option to clear it', async () => {
  vi.mocked(invoke).mockResolvedValueOnce({
    base_url: 'http://127.0.0.1:8766', ui_token: crypto.randomUUID(),
    siliconflow_restore_error: 'KEYCHAIN_CREDENTIAL_CORRUPTED',
  }).mockResolvedValueOnce({ configured: false });
  vi.stubGlobal('fetch', (url: string) => Promise.resolve(Response.json(url.endsWith('/sessions') ? [] : {
    status: 'ready', database: 'ready', model_configured: true, siliconflow_configured: false,
  })));

  render(<App />);
  const error = 'macOS 钥匙串中的 SiliconFlow 凭据已损坏，请重新保存或清除。';
  await screen.findByText(error);
  expect(screen.getByText('已配置')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '清除 Key' }));
  await screen.findByText('已删除 SiliconFlow 凭据。');
  expect(screen.queryByText(error)).toBeNull();
});

test('shows the model as unconfigured if credentials cannot be restored', async () => {
  vi.mocked(invoke).mockRejectedValue('MODEL_CREDENTIAL_REPLAY_FAILED');
  render(<App />);
  expect(await screen.findByText('DeepSeek 凭据恢复失败，模型未配置。可重试连接，或退出应用后重新配置。')).toBeTruthy();
  expect(screen.queryByText('已配置')).toBeNull();
});

test('updates and deletes Keychain credentials and reports Keychain errors without leaking secrets', async () => {
  const initialKey = crypto.randomUUID();
  const rotatedKey = crypto.randomUUID();
  vi.mocked(invoke)
    .mockResolvedValueOnce({ base_url: 'http://127.0.0.1:8766', ui_token: crypto.randomUUID() })
    .mockResolvedValueOnce({ configured: true })
    .mockResolvedValueOnce({ configured: true })
    .mockRejectedValueOnce('KEYCHAIN_ACCESS_FAILED')
    .mockResolvedValueOnce({ configured: false });
  vi.stubGlobal('fetch', (url: string) =>
    url.endsWith('/sessions')
      ? Promise.resolve(Response.json([]))
      : Promise.resolve(Response.json({ status: 'ready', database: 'ready', model_configured: false })));

  render(<App />);
  const field = await screen.findByLabelText('DeepSeek API Key') as HTMLInputElement;
  expect(screen.getByText('未配置')).toBeTruthy();
  expect(screen.queryByRole('button', { name: '删除凭据' })).toBeNull();

  fireEvent.change(field, { target: { value: initialKey } });
  fireEvent.click(screen.getByRole('button', { name: '配置' }));
  expect(await screen.findByText('凭据已提交。')).toBeTruthy();
  expect(screen.getByText('已配置')).toBeTruthy();
  expect(screen.getByRole('button', { name: '更新' })).toBeTruthy();

  fireEvent.change(field, { target: { value: rotatedKey } });
  fireEvent.click(screen.getByRole('button', { name: '更新' }));
  expect(invoke).toHaveBeenLastCalledWith('save_provider_credential', { provider: 'deepseek', apiKey: rotatedKey });
  expect(await screen.findByText('凭据已提交。')).toBeTruthy();

  const deleteButton = screen.getByRole('button', { name: '删除凭据' });
  fireEvent.click(deleteButton);
  expect(invoke).toHaveBeenLastCalledWith('delete_provider_credential', { provider: 'deepseek' });
  expect(await screen.findByText('无法访问 macOS Keychain，未能删除 DeepSeek 凭据。')).toBeTruthy();
  expect(screen.getByText('已配置')).toBeTruthy();

  fireEvent.click(deleteButton);
  expect(await screen.findByText('已删除 DeepSeek 凭据。')).toBeTruthy();
  expect(screen.getByText('未配置')).toBeTruthy();
  expect(screen.queryByRole('button', { name: '删除凭据' })).toBeNull();
});

test.each([
  ['KEYCHAIN_ACCESS_FAILED', '无法访问 macOS Keychain 读取 DeepSeek 凭据，请检查钥匙串权限后重试。'],
  ['KEYCHAIN_CREDENTIAL_CORRUPTED', 'macOS Keychain 中的 DeepSeek 凭据已损坏，请清理或重新配置。'],
  ['BACKEND_DATABASE_LOCKED', '检测到另一个 KnowledgeAgent 实例正在运行并占用数据目录，请先关闭已运行的应用后重试。'],
] as const)('displays explicit startup error when backend connection fails with %s', async (code, expected) => {
  vi.mocked(invoke).mockRejectedValue(code);
  render(<App />);
  expect(await screen.findByText(expected)).toBeTruthy();
  expect(screen.queryByText('已配置')).toBeNull();
});

test('opens the application log directory through restricted IPC and reports errors clearly', async () => {
  vi.mocked(invoke)
    .mockResolvedValueOnce({ base_url: 'http://127.0.0.1:8766', ui_token: crypto.randomUUID() })
    .mockRejectedValueOnce('LOG_DIRECTORY_OPEN_FAILED')
    .mockResolvedValueOnce(undefined);
  vi.stubGlobal('fetch', (url: string) =>
    url.endsWith('/sessions')
      ? Promise.resolve(Response.json([]))
      : Promise.resolve(Response.json({ status: 'ready', database: 'ready', model_configured: false })));

  render(<App />);
  expect(await screen.findByText('ready')).toBeTruthy();

  const logButton = screen.getByRole('button', { name: '打开日志目录' });
  fireEvent.click(logButton);
  expect(invoke).toHaveBeenLastCalledWith('open_log_directory');
  expect(await screen.findByText('无法打开日志目录，请检查应用数据目录权限。')).toBeTruthy();

  fireEvent.click(logButton);
  expect(invoke).toHaveBeenLastCalledWith('open_log_directory');
  await screen.findByRole('button', { name: '打开日志目录' });
  expect(screen.queryByText('无法打开日志目录，请检查应用数据目录权限。')).toBeNull();
});
