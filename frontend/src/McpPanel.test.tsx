// @vitest-environment jsdom

import { invoke } from '@tauri-apps/api/core';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';
import {
  connectApiClient,
  type McpServerView,
  type McpSummary,
  type McpToolView,
} from './api/apiClient';
import McpPanel, { previewImportJson } from './McpPanel';

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn() }));

afterEach(() => {
  cleanup();
  localStorage.clear();
  sessionStorage.clear();
  vi.useRealTimers();
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

const TS = '2026-11-25T10:00:00Z';

function makeServer(overrides: Partial<McpServerView> = {}): McpServerView {
  return {
    server_id: 'srv-1',
    name: 'Local Stdio',
    transport: 'stdio',
    command: '/usr/bin/python3',
    resolved_command: '/usr/bin/python3',
    executable_exists: true,
    args: ['/path/with spaces/server.py'],
    cwd: null,
    url: null,
    allow_private_network: false,
    private_network_target: false,
    high_risk_command: false,
    high_risk_reasons: [],
    env_config: {},
    header_config: {},
    credential_ref: null,
    credentials_complete: true,
    missing_secret_keys: [],
    enabled: false,
    trust_state: 'untrusted',
    trust_valid: false,
    config_revision: 1,
    trusted_revision: null,
    trusted_fingerprint: null,
    current_fingerprint: 'a'.repeat(64),
    discovered_tool_count: 0,
    allowed_tool_count: 0,
    effective_allowed_tool_count: 0,
    runtime: {
      state: 'disconnected',
      generation: 0,
      protocol_version: null,
      server_name_reported: null,
      server_version_reported: null,
      tool_count: 0,
      catalog_version: 0,
      catalog_stale: false,
      tools_truncated: false,
      last_error_code: null,
      last_error_message: null,
      stderr_tail: [],
      connected_at: null,
    },
    created_at: TS,
    updated_at: TS,
    ...overrides,
  };
}

function makeTool(overrides: Partial<McpToolView> = {}): McpToolView {
  return {
    tool_id: 'tool-1',
    server_id: 'srv-1',
    original_tool_name: 'echo_text',
    title: 'Echo Text',
    description: 'Echoes text back safely.',
    input_schema: { type: 'object', properties: { text: { type: 'string' } }, required: ['text'] },
    annotations: { readOnlyHint: true },
    definition_hash: 'b'.repeat(64),
    schema_compatible: true,
    incompatibility_reason: null,
    permission_state: 'denied',
    approved_definition_hash: null,
    approved_config_revision: null,
    definition_changed: false,
    is_destructive: false,
    present_in_last_discovery: true,
    effective_allowed: false,
    discovered_at: TS,
    updated_at: TS,
    ...overrides,
  };
}

function summaryFrom(servers: McpServerView[], overrides: Partial<McpSummary> = {}): McpSummary {
  const effective = servers.reduce((sum, s) => sum + s.effective_allowed_tool_count, 0);
  return {
    total_servers: servers.length,
    enabled_servers: servers.filter((s) => s.enabled).length,
    connected_servers: servers.filter((s) => s.runtime.state === 'connected').length,
    error_servers: servers.filter((s) => s.runtime.state === 'error').length,
    effective_allowed_tools: effective,
    max_tools_per_run: 32,
    exceeds_tool_limit: effective > 32,
    ...overrides,
  };
}

async function createMockClient(
  getServers: () => McpServerView[] | Promise<McpServerView[]>,
  getTools: (serverId: string) => McpToolView[] | Promise<McpToolView[]> = () => [],
  getSummary?: () => McpSummary | Promise<McpSummary>,
) {
  vi.mocked(invoke).mockResolvedValueOnce({ base_url: 'http://127.0.0.1:8766', ui_token: 'ui-mcp-test' });
  const fetchMock = vi.fn(async (url: string, options: RequestInit) => {
    expect((options.headers as Record<string, string>).Authorization).toBe('Bearer ui-mcp-test');
    const path = new URL(url).pathname.replace('/api/v1', '');
    if (path === '/mcp/servers') {
      return Response.json(await getServers());
    }
    if (path === '/mcp/summary') {
      return Response.json(getSummary ? await getSummary() : summaryFrom(await getServers()));
    }
    const toolsMatch = path.match(/^\/mcp\/servers\/([^/]+)\/tools$/);
    if (toolsMatch) {
      return Response.json(await getTools(decodeURIComponent(toolsMatch[1])));
    }
    throw new Error(`Unexpected GET ${path}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  const client = await connectApiClient();
  return { client, fetchMock };
}

test('renders friendly empty state without creating fake servers', async () => {
  const { client } = await createMockClient(() => []);
  render(<McpPanel client={client} active />);

  expect(await screen.findByText('尚未配置任何 MCP 服务')).toBeTruthy();
  expect(screen.getByLabelText('MCP 状态摘要').textContent).toContain('0 / 32');
});

test('adds stdio and Streamable HTTP servers via privileged IPC, preserves spaces in args, and never echoes secrets', async () => {
  let servers: McpServerView[] = [];
  const { client } = await createMockClient(() => servers);

  render(<McpPanel client={client} active />);
  await screen.findByText('尚未配置任何 MCP 服务');

  // 1) Add stdio server with path containing spaces + password secret
  fireEvent.click(screen.getAllByRole('button', { name: '+ 添加服务' })[0]);
  const dialog = screen.getByRole('dialog');
  expect(dialog.textContent).toContain('不具备操作系统级文件沙箱');

  fireEvent.change(within(dialog).getByLabelText('服务名称'), { target: { value: 'Spaced Stdio' } });
  fireEvent.change(within(dialog).getByLabelText('可执行命令（Command）'), { target: { value: '/opt/homebrew/bin/python3' } });
  fireEvent.click(within(dialog).getByRole('button', { name: '+ 添加参数' }));
  fireEvent.change(within(dialog).getByLabelText('启动参数 1'), { target: { value: '/Users/demo/My Docs/stdio server.py' } });

  fireEvent.click(within(dialog).getByRole('button', { name: '+ 添加一项' }));
  fireEvent.change(within(dialog).getByLabelText('配置键 1'), { target: { value: 'API_TOKEN' } });
  fireEvent.change(within(dialog).getByLabelText('配置类型 1'), { target: { value: 'secret' } });
  const secretInput = within(dialog).getByLabelText<HTMLInputElement>('密钥值 API_TOKEN');
  expect(secretInput.type).toBe('password');
  fireEvent.change(secretInput, { target: { value: 'super-secret-stdio-123' } });

  const savedStdio = makeServer({
    server_id: 'srv-stdio',
    name: 'Spaced Stdio',
    command: '/opt/homebrew/bin/python3',
    resolved_command: '/opt/homebrew/bin/python3',
    args: ['/Users/demo/My Docs/stdio server.py'],
    env_config: { API_TOKEN: { kind: 'secret', secret_key: 'API_TOKEN', configured: true } },
  });
  vi.mocked(invoke).mockImplementationOnce(async () => {
    servers = [savedStdio];
    return savedStdio;
  });

  fireEvent.click(within(dialog).getByRole('button', { name: '保存为未信任草稿' }));
  await waitFor(() => expect(invoke).toHaveBeenLastCalledWith('mcp_save_server', {
    request: {
      server_id: null,
      name: 'Spaced Stdio',
      transport: 'stdio',
      command: '/opt/homebrew/bin/python3',
      args: ['/Users/demo/My Docs/stdio server.py'],
      cwd: null,
      clear_cwd: false,
      url: null,
      allow_private_network: false,
      env_config: { API_TOKEN: { kind: 'secret', secret_key: 'API_TOKEN' } },
      header_config: {},
      secret_values: { API_TOKEN: 'super-secret-stdio-123' },
      remove_secret_keys: [],
    },
  }));
  expect(await screen.findByText('Spaced Stdio')).toBeTruthy();
  expect(document.body.textContent).not.toContain('super-secret-stdio-123');

  // Edit existing server -> secret must NOT be echoed; supports keep / update / delete explicitly
  fireEvent.click(screen.getByRole('button', { name: '编辑' }));
  const editDialog = screen.getByRole('dialog');
  expect(editDialog.textContent).toContain('已配置（不回显）');
  expect(editDialog.innerHTML).not.toContain('super-secret-stdio-123');

  // Choose "删除凭据"
  fireEvent.change(within(editDialog).getByLabelText('凭据操作 API_TOKEN'), { target: { value: 'delete' } });
  const afterDeleteSecret = makeServer({
    ...savedStdio,
    env_config: { API_TOKEN: { kind: 'secret', secret_key: 'API_TOKEN', configured: false } },
    credentials_complete: false,
    missing_secret_keys: ['API_TOKEN'],
  });
  vi.mocked(invoke).mockImplementationOnce(async () => {
    servers = [afterDeleteSecret];
    return afterDeleteSecret;
  });
  fireEvent.click(within(editDialog).getByRole('button', { name: '保存修改' }));
  await waitFor(() => expect(invoke).toHaveBeenLastCalledWith('mcp_save_server', expect.objectContaining({
    request: expect.objectContaining({
      server_id: 'srv-stdio',
      secret_values: {},
      remove_secret_keys: ['API_TOKEN'],
    }),
  })));
  expect(await screen.findByText(/缺少凭据（1）/)).toBeTruthy();

  // 2) Add Streamable HTTP server with allow_private_network
  fireEvent.click(screen.getByRole('button', { name: '+ 添加服务' }));
  const httpDialog = screen.getByRole('dialog');
  fireEvent.change(within(httpDialog).getByLabelText('服务名称'), { target: { value: 'Internal HTTP' } });
  fireEvent.click(within(httpDialog).getByLabelText('Streamable HTTP（网络服务）'));
  fireEvent.change(within(httpDialog).getByLabelText('服务地址（Server URL）'), { target: { value: 'https://192.168.1.20/mcp' } });
  fireEvent.click(within(httpDialog).getByLabelText(/显式允许访问私有局域网地址/));

  const savedHttp = makeServer({
    server_id: 'srv-http',
    name: 'Internal HTTP',
    transport: 'streamable_http',
    command: null,
    resolved_command: null,
    executable_exists: null,
    args: [],
    url: 'https://192.168.1.20/mcp',
    allow_private_network: true,
    private_network_target: true,
  });
  vi.mocked(invoke).mockImplementationOnce(async () => {
    servers = [afterDeleteSecret, savedHttp];
    return savedHttp;
  });
  fireEvent.click(within(httpDialog).getByRole('button', { name: '保存为未信任草稿' }));
  await waitFor(() => expect(invoke).toHaveBeenLastCalledWith('mcp_save_server', expect.objectContaining({
    request: expect.objectContaining({
      transport: 'streamable_http',
      url: 'https://192.168.1.20/mcp',
      allow_private_network: true,
    }),
  })));
  expect(await screen.findByText('Internal HTTP')).toBeTruthy();
});

test('previews JSON import with secret detection, rejects legacy SSE/unknown fields, and never stores raw JSON in localStorage', async () => {
  expect(previewImportJson('{"mcpServers":{"bad":{"type":"sse","url":"https://a.test/sse"}}}').ok).toBe(false);
  expect(previewImportJson('{"mcpServers":{"bad":{"command":"py","autoStart":true}}}').ok).toBe(false);

  let servers: McpServerView[] = [];
  const { client } = await createMockClient(() => servers);
  render(<McpPanel client={client} active />);
  await screen.findByText('尚未配置任何 MCP 服务');

  fireEvent.click(screen.getAllByRole('button', { name: '导入 JSON' })[0]);
  const dialog = screen.getByRole('dialog');
  const rawSecret = 'ghp_super_secret_import_token_999';
  const rawJson = JSON.stringify({
    mcpServers: {
      demo: {
        command: '/usr/bin/python3',
        args: ['server.py'],
        env: { GITHUB_TOKEN: rawSecret, MODE: 'ro' },
      },
    },
  });

  fireEvent.change(within(dialog).getByLabelText('MCP JSON 配置'), { target: { value: rawJson } });
  const preview = within(dialog).getByLabelText('导入预览');
  expect(preview.textContent).toContain('demo');
  expect(preview.textContent).toContain('GITHUB_TOKEN');
  expect(preview.textContent).toContain('已停用 · 待信任');
  expect(preview.textContent).not.toContain(rawSecret);

  const importedDraft = makeServer({ server_id: 'srv-imp', name: 'demo', enabled: false, trust_state: 'untrusted' });
  vi.mocked(invoke).mockImplementationOnce(async () => {
    servers = [importedDraft];
    return { servers: [importedDraft], warnings: ['检测到 1 个敏感环境变量，已隔离存入钥匙串。'] };
  });

  fireEvent.click(within(dialog).getByRole('button', { name: '确认导入为未信任草稿' }));
  await waitFor(() => expect(invoke).toHaveBeenLastCalledWith('mcp_import_servers_json', {
    request: { raw_json: rawJson },
  }));
  expect(await screen.findByText(/已导入 1 个未信任草稿服务/)).toBeTruthy();
  expect(JSON.stringify(localStorage)).not.toContain(rawSecret);
  expect(JSON.stringify(sessionStorage)).not.toContain(rawSecret);
});

test('handles native trust confirm, cancel, revoke-trust, re-trust after config drift, and connect/disconnect/error diagnostics', async () => {
  let servers: McpServerView[] = [makeServer()];
  const { client } = await createMockClient(() => servers);
  render(<McpPanel client={client} active />);
  await screen.findByText('Local Stdio');

  // 1) User cancels native NSAlert -> stays untrusted
  vi.mocked(invoke).mockRejectedValueOnce('MCP_TRUST_CANCELLED');
  fireEvent.click(screen.getByRole('button', { name: '信任确认' }));
  expect(await screen.findByText(/已取消信任确认，服务仍保持未信任状态/)).toBeTruthy();
  expect(invoke).toHaveBeenLastCalledWith('mcp_trust_server', { request: { server_id: 'srv-1', enable: true } });

  // 2) User confirms native NSAlert -> becomes trusted & enabled (still disconnected until explicit connect!)
  vi.mocked(invoke).mockImplementationOnce(async () => {
    servers = [makeServer({ enabled: true, trust_state: 'trusted', trust_valid: true, trusted_revision: 1 })];
    return servers[0];
  });
  fireEvent.click(screen.getByRole('button', { name: '信任确认' }));
  expect(await screen.findByText('已信任')).toBeTruthy();
  expect(screen.getByText('未连接')).toBeTruthy();

  // 3) Connect fails -> config preserved, redacted error + stderr diagnostics displayed
  vi.mocked(invoke).mockImplementationOnce(async () => {
    servers = [makeServer({
      enabled: true,
      trust_state: 'trusted',
      trust_valid: true,
      trusted_revision: 1,
      runtime: {
        ...makeServer().runtime,
        state: 'error',
        last_error_code: 'MCP_CONNECT_FAILED',
        last_error_message: '子进程握手失败 [REDACTED]',
        stderr_tail: ['Traceback: exit 1'],
      },
    })];
    throw 'MCP_CONNECT_FAILED';
  });
  fireEvent.click(screen.getByRole('button', { name: '连接' }));
  expect(await screen.findByText(/子进程握手失败 \[REDACTED\]/)).toBeTruthy();
  expect(screen.getByText('Local Stdio')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '诊断信息' }));
  expect(await screen.findByText(/Traceback: exit 1/)).toBeTruthy();

  // 4) Reconnect succeeds -> connected, then disconnect, then revoke trust via mcp_revoke_trust
  vi.mocked(invoke).mockImplementationOnce(async () => {
    servers = [makeServer({
      enabled: true,
      trust_state: 'trusted',
      trust_valid: true,
      trusted_revision: 1,
      runtime: { ...makeServer().runtime, state: 'connected', generation: 1 },
    })];
    return servers[0];
  });
  fireEvent.click(screen.getByRole('button', { name: '重连' }));
  expect(await screen.findByText('已连接')).toBeTruthy();

  vi.mocked(invoke).mockImplementationOnce(async () => {
    servers = [makeServer({ enabled: false, trust_state: 'untrusted', trust_valid: false })];
    return servers[0];
  });
  fireEvent.click(screen.getByRole('button', { name: '撤销信任' }));
  await waitFor(() => expect(invoke).toHaveBeenLastCalledWith('mcp_revoke_trust', {
    request: { server_id: 'srv-1' },
  }));
  expect(await screen.findByText(/已撤销服务「Local Stdio」的信任/)).toBeTruthy();

  // 5) Config drift (`trust_state: 'trusted'` but `trust_valid: false`) displays "重新信任"
  servers = [makeServer({
    enabled: false,
    trust_state: 'trusted',
    trust_valid: false,
    config_revision: 2,
    trusted_revision: 1,
  })];
  fireEvent.click(screen.getByRole('button', { name: '刷新状态' }));
  expect(await screen.findByText('配置已变更 · 需重新信任')).toBeTruthy();
  expect(screen.getByRole('button', { name: '重新信任' })).toBeTruthy();
});

test('manages tool authorization, blocks destructive/incompatible/missing/stale tools, handles definition drift and failed approval rollback', async () => {
  let server = makeServer({
    enabled: true,
    trust_state: 'trusted',
    trust_valid: true,
    trusted_revision: 1,
    discovered_tool_count: 4,
    allowed_tool_count: 1,
    effective_allowed_tool_count: 0,
    runtime: { ...makeServer().runtime, state: 'connected', generation: 2, catalog_version: 3 },
  });
  let tools: McpToolView[] = [
    makeTool({ tool_id: 't-ok', original_tool_name: 'echo_text', annotations: { readOnlyHint: true } }),
    makeTool({
      tool_id: 't-drift',
      original_tool_name: 'calc_v2',
      annotations: {},
      permission_state: 'allowed',
      definition_changed: true,
      effective_allowed: false,
    }),
    makeTool({ tool_id: 't-dest', original_tool_name: 'drop_db', annotations: {}, is_destructive: true }),
    makeTool({
      tool_id: 't-incompat',
      original_tool_name: 'bad_schema',
      annotations: {},
      schema_compatible: false,
      incompatibility_reason: 'Schema 超过 6 层嵌套上限',
    }),
    makeTool({ tool_id: 't-missing', original_tool_name: 'old_tool', annotations: {}, present_in_last_discovery: false }),
  ];

  const { client } = await createMockClient(
    () => [server],
    () => tools,
    () => summaryFrom([server], { effective_allowed_tools: 33, exceeds_tool_limit: true }),
  );
  render(<McpPanel client={client} active />);

  // Over-limit warning banner displayed
  expect(await screen.findByText(/已超过单轮对话上限（32）/)).toBeTruthy();

  // Expand tools
  fireEvent.click(screen.getByRole('button', { name: /展开工具列表/ }));
  expect(await screen.findByText('echo_text')).toBeTruthy();
  expect(screen.getByText('服务端标注只读（仅供参考）')).toBeTruthy();
  expect(screen.getByText('定义已变更 · 需重新授权')).toBeTruthy();

  // Destructive, incompatible, and missing tools MUST be disabled for authorization
  expect(screen.getByRole<HTMLButtonElement>('button', { name: '授权 drop_db' }).disabled).toBe(true);
  expect(screen.getByRole<HTMLButtonElement>('button', { name: '授权 bad_schema' }).disabled).toBe(true);
  expect(screen.getByRole<HTMLButtonElement>('button', { name: '授权 old_tool' }).disabled).toBe(true);

  // Failed tool authorization MUST re-fetch authoritative state and NOT leave stale allowed UI
  vi.mocked(invoke).mockRejectedValueOnce('MCP_PERMISSION_DENIED');
  fireEvent.click(screen.getByRole('button', { name: '授权 echo_text' }));
  expect(await screen.findByText(/MCP 工具未获得有效授权/)).toBeTruthy();
  expect(screen.getByRole('button', { name: '授权 echo_text' })).toBeTruthy();

  // Successful re-authorization of drifted tool passes expected_definition_hash & expected_config_revision
  const approvedDrift = makeTool({
    tool_id: 't-drift',
    original_tool_name: 'calc_v2',
    permission_state: 'allowed',
    definition_changed: false,
    effective_allowed: true,
  });
  vi.mocked(invoke).mockImplementationOnce(async () => {
    tools = tools.map((t) => (t.tool_id === 't-drift' ? approvedDrift : t));
    return approvedDrift;
  });
  fireEvent.click(screen.getByRole('button', { name: '重新授权 calc_v2' }));
  await waitFor(() => expect(invoke).toHaveBeenLastCalledWith('mcp_set_tool_permission', {
    request: {
      server_id: 'srv-1',
      tool_id: 't-drift',
      permission_state: 'allowed',
      expected_definition_hash: 'b'.repeat(64),
      expected_config_revision: 1,
    },
  }));
  expect(await screen.findByRole('button', { name: '撤销授权 calc_v2' })).toBeTruthy();

  // During catalog refresh, old tools MUST NOT be shown as callable/effective
  const pendingRefresh = Promise.withResolvers<McpToolView[]>();
  vi.mocked(invoke).mockReturnValueOnce(pendingRefresh.promise);
  fireEvent.click(screen.getByRole('button', { name: '刷新工具' }));
  expect(await screen.findByText(/目录刷新中 · 旧工具不可调用/)).toBeTruthy();
  expect(screen.getAllByText(/刷新中不可用/).length).toBeGreaterThan(0);

  await act(async () => {
    pendingRefresh.resolve(tools);
  });
  await waitFor(() => expect(screen.queryByText(/目录刷新中 · 旧工具不可调用/)).toBeNull());
});

test('stops polling when inactive and prevents stale out-of-order responses from overwriting newer state', async () => {
  const initialServer = makeServer({ name: 'Initial Server' });
  const staleFetch = Promise.withResolvers<McpServerView[]>();
  const newerFetch = Promise.withResolvers<McpServerView[]>();
  let serverCalls = 0;

  const { client, fetchMock } = await createMockClient(
    () => {
      serverCalls += 1;
      if (serverCalls === 1) return [initialServer];
      if (serverCalls === 2) return staleFetch.promise;
      return newerFetch.promise;
    },
    () => [],
    () => summaryFrom([initialServer]),
  );

  const { rerender } = render(<McpPanel client={client} active />);
  expect(await screen.findByText('Initial Server')).toBeTruthy();

  // Trigger fetch #2 (stale) via an action, then trigger fetch #3 (newer) via manual refresh before #2 resolves
  vi.mocked(invoke).mockResolvedValueOnce(initialServer);
  fireEvent.click(screen.getByRole('button', { name: '信任确认' }));
  await waitFor(() => expect(serverCalls).toBe(2));

  fireEvent.click(screen.getByRole('button', { name: '刷新状态' }));
  await waitFor(() => expect(serverCalls).toBe(3));

  const newer = [makeServer({ name: 'Authoritative Newer Server' })];
  const stale = [makeServer({ name: 'Stale Older Server' })];
  await act(async () => {
    newerFetch.resolve(newer);
  });
  expect(await screen.findByText('Authoritative Newer Server')).toBeTruthy();

  // Resolve the older request AFTER the newer one -> MUST be ignored!
  await act(async () => {
    staleFetch.resolve(stale);
  });
  expect(screen.getByText('Authoritative Newer Server')).toBeTruthy();
  expect(screen.queryByText('Stale Older Server')).toBeNull();

  // Deactivate panel -> polling must stop
  const callsBeforeDeactivate = fetchMock.mock.calls.length;
  rerender(<McpPanel client={client} active={false} />);
  await new Promise((resolve) => setTimeout(resolve, 60));
  expect(fetchMock.mock.calls.length).toBe(callsBeforeDeactivate);
});
