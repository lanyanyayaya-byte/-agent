import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import {
  mcpConnectServer,
  mcpDeleteServer,
  mcpDisconnectServer,
  mcpErrorMessage,
  mcpImportServersJson,
  mcpRefreshTools,
  mcpRevokeTrust,
  mcpSaveServer,
  mcpSetServerEnabled,
  mcpSetToolPermission,
  mcpTrustServer,
  type ApiClient,
  type McpConfigEntryRequest,
  type McpRuntimeState,
  type McpServerView,
  type McpSummary,
  type McpToolView,
  type McpTransportKind,
  type SaveMcpServerRequest,
} from './api/apiClient';

const MCP_POLL_INTERVAL_MS = 3000;
const MAX_IMPORT_JSON_BYTES = 256 * 1024;
const MAX_IMPORT_SERVERS = 16;

type EntryDraft = {
  id: string;
  key: string;
  kind: 'plain' | 'secret';
  value: string;
  existingConfigured: boolean;
  secretAction: 'keep' | 'update' | 'delete';
};

type ArgDraft = {
  id: string;
  value: string;
};

type ServerFormState = {
  serverId: string | null;
  name: string;
  transport: McpTransportKind;
  command: string;
  args: ArgDraft[];
  cwd: string;
  hadCwd: boolean;
  url: string;
  allowPrivateNetwork: boolean;
  envEntries: EntryDraft[];
  headerEntries: EntryDraft[];
  originalSecretKeys: string[];
};

export type ImportPreviewServer = {
  name: string;
  transport: McpTransportKind;
  summaryTarget: string;
  args: string[];
  cwd?: string;
  allowPrivateNetwork?: boolean;
  plainKeys: string[];
  secretKeys: string[];
};

export type ImportPreviewResult =
  | { ok: true; servers: ImportPreviewServer[] }
  | { ok: false; error: string };

let draftCounter = 0;
function nextDraftId(prefix: string): string {
  draftCounter += 1;
  return `${prefix}-${draftCounter}`;
}

function isSuspectedSecretKey(key: string): boolean {
  const upper = key.trim().toUpperCase();
  return ['KEY', 'TOKEN', 'SECRET', 'AUTH', 'PASSWORD', 'PASSWD', 'CREDENTIAL', 'COOKIE', 'PRIVATE']
    .some((part) => upper.includes(part));
}

function isStrictSecretHeader(key: string): boolean {
  return [
    'authorization',
    'proxy-authorization',
    'x-api-key',
    'api-key',
    'x-auth-token',
    'x-access-token',
    'cookie',
  ].includes(key.trim().toLowerCase());
}

function looksLikeSecretValue(val: string): boolean {
  const lower = val.trim().toLowerCase();
  return lower.startsWith('bearer ')
    || lower.startsWith('basic ')
    || val.includes('sk-')
    || val.includes('ghp_')
    || val.includes('github_pat_')
    || val.startsWith('eyJ');
}

/**
 * Pure client-side preview parser mirroring desktop/src/mcp.rs `parse_import_json`.
 * Never executes commands, never persists raw JSON, and never echoes secret values.
 */
export function previewImportJson(rawJson: string): ImportPreviewResult {
  const trimmed = rawJson.trim();
  if (!trimmed) {
    return { ok: false, error: '请粘贴有效的 MCP JSON 配置。' };
  }
  if (new TextEncoder().encode(rawJson).byteLength > MAX_IMPORT_JSON_BYTES) {
    return { ok: false, error: 'JSON 体积超过 256 KiB 上限。' };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawJson);
  } catch {
    return { ok: false, error: 'JSON 语法无效，请检查括号、引号与逗号格式。' };
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { ok: false, error: '顶层必须是包含 "mcpServers" 字段的 JSON 对象。' };
  }
  const rootObj = parsed as Record<string, unknown>;
  const rootKeys = Object.keys(rootObj);
  if (rootKeys.length !== 1 || !('mcpServers' in rootObj)) {
    return { ok: false, error: '顶层仅允许包含 "mcpServers" 字段，请移除多余的未知字段。' };
  }
  const serversRaw = rootObj.mcpServers;
  if (!serversRaw || typeof serversRaw !== 'object' || Array.isArray(serversRaw)) {
    return { ok: false, error: '"mcpServers" 必须是以服务名为键的 JSON 对象。' };
  }
  const entries = Object.entries(serversRaw as Record<string, unknown>);
  if (entries.length === 0) {
    return { ok: false, error: '"mcpServers" 中至少需要包含 1 个服务。' };
  }
  if (entries.length > MAX_IMPORT_SERVERS) {
    return { ok: false, error: `单次最多导入 ${MAX_IMPORT_SERVERS} 个 MCP 服务。` };
  }

  const seenNames = new Set<string>();
  const servers: ImportPreviewServer[] = [];

  for (const [rawName, val] of entries) {
    const name = rawName.trim();
    if (!name || name.length > 100) {
      return { ok: false, error: `服务名称「${rawName}」无效（需 1–100 字符）。` };
    }
    const lowerName = name.toLowerCase();
    if (seenNames.has(lowerName)) {
      return { ok: false, error: `JSON 中存在重复的服务名称「${name}」。` };
    }
    seenNames.add(lowerName);

    if (!val || typeof val !== 'object' || Array.isArray(val)) {
      return { ok: false, error: `服务「${name}」的配置必须是 JSON 对象。` };
    }
    const entry = val as Record<string, unknown>;
    const explicitTransport = typeof entry.transport === 'string'
      ? entry.transport
      : typeof entry.type === 'string'
        ? entry.type
        : undefined;

    let transport: McpTransportKind;
    if (explicitTransport !== undefined) {
      const normalized = explicitTransport.trim().toLowerCase();
      if (normalized === 'stdio') {
        transport = 'stdio';
      } else if (['http', 'streamable-http', 'streamable_http'].includes(normalized)) {
        transport = 'streamable_http';
      } else {
        return { ok: false, error: `服务「${name}」使用了不支持的传输类型「${explicitTransport}」（仅支持 stdio 与 Streamable HTTP）。` };
      }
    } else if ('command' in entry && !('url' in entry)) {
      transport = 'stdio';
    } else if ('url' in entry && !('command' in entry)) {
      transport = 'streamable_http';
    } else {
      return { ok: false, error: `服务「${name}」必须明确指定 command（stdio）或 url（Streamable HTTP），二者不可同时填写。` };
    }

    const plainKeys: string[] = [];
    const secretKeys: string[] = [];

    if (transport === 'stdio') {
      const allowedKeys = new Set(['command', 'args', 'cwd', 'env', 'transport', 'type']);
      for (const k of Object.keys(entry)) {
        if (!allowedKeys.has(k)) {
          return { ok: false, error: `服务「${name}」包含不支持的字段「${k}」。` };
        }
      }
      if (typeof entry.command !== 'string' || !entry.command.trim()) {
        return { ok: false, error: `stdio 服务「${name}」缺少有效的 command 字段。` };
      }
      const args: string[] = [];
      if (entry.args !== undefined) {
        if (!Array.isArray(entry.args) || !entry.args.every((item) => typeof item === 'string')) {
          return { ok: false, error: `服务「${name}」的 args 必须是字符串数组。` };
        }
        args.push(...entry.args);
      }
      let cwd: string | undefined;
      if (entry.cwd !== undefined && entry.cwd !== null) {
        if (typeof entry.cwd !== 'string') {
          return { ok: false, error: `服务「${name}」的 cwd 必须是字符串。` };
        }
        cwd = entry.cwd;
      }
      if (entry.env !== undefined) {
        if (!entry.env || typeof entry.env !== 'object' || Array.isArray(entry.env)) {
          return { ok: false, error: `服务「${name}」的 env 必须是字符串键值对对象。` };
        }
        for (const [k, v] of Object.entries(entry.env as Record<string, unknown>)) {
          if (typeof v !== 'string') {
            return { ok: false, error: `服务「${name}」环境变量「${k}」的值必须是字符串。` };
          }
          if (isSuspectedSecretKey(k) || looksLikeSecretValue(v)) {
            secretKeys.push(k);
          } else {
            plainKeys.push(k);
          }
        }
      }
      servers.push({
        name,
        transport,
        summaryTarget: entry.command.trim(),
        args,
        cwd,
        plainKeys,
        secretKeys,
      });
    } else {
      const allowedKeys = new Set(['url', 'headers', 'transport', 'type', 'allowPrivateNetwork', 'allow_private_network']);
      for (const k of Object.keys(entry)) {
        if (!allowedKeys.has(k)) {
          return { ok: false, error: `HTTP 服务「${name}」包含不支持的字段「${k}」。` };
        }
      }
      if (typeof entry.url !== 'string' || !entry.url.trim()) {
        return { ok: false, error: `HTTP 服务「${name}」缺少有效的 url 字段。` };
      }
      const allowPrivate = typeof entry.allowPrivateNetwork === 'boolean'
        ? entry.allowPrivateNetwork
        : typeof entry.allow_private_network === 'boolean'
          ? entry.allow_private_network
          : false;
      if (entry.headers !== undefined) {
        if (!entry.headers || typeof entry.headers !== 'object' || Array.isArray(entry.headers)) {
          return { ok: false, error: `服务「${name}」的 headers 必须是字符串键值对对象。` };
        }
        for (const [k, v] of Object.entries(entry.headers as Record<string, unknown>)) {
          if (typeof v !== 'string') {
            return { ok: false, error: `服务「${name}」请求头「${k}」的值必须是字符串。` };
          }
          if (isStrictSecretHeader(k) || isSuspectedSecretKey(k) || looksLikeSecretValue(v)) {
            secretKeys.push(k);
          } else {
            plainKeys.push(k);
          }
        }
      }
      servers.push({
        name,
        transport,
        summaryTarget: entry.url.trim(),
        args: [],
        allowPrivateNetwork: allowPrivate,
        plainKeys,
        secretKeys,
      });
    }
  }

  return { ok: true, servers };
}

function runtimeLabel(state: McpRuntimeState): string {
  switch (state) {
    case 'connected': return '已连接';
    case 'connecting': return '连接中';
    case 'disconnecting': return '断开中';
    case 'error': return '连接异常';
    default: return '未连接';
  }
}

function trustBadgeInfo(server: McpServerView): { label: string; tone: 'ok' | 'warn' | 'muted' } {
  if (server.trust_valid) {
    return { label: '已信任', tone: 'ok' };
  }
  if (server.trust_state === 'trusted') {
    return { label: '配置已变更 · 需重新信任', tone: 'warn' };
  }
  return { label: '待信任', tone: 'muted' };
}

function createEmptyForm(): ServerFormState {
  return {
    serverId: null,
    name: '',
    transport: 'stdio',
    command: '',
    args: [],
    cwd: '',
    hadCwd: false,
    url: '',
    allowPrivateNetwork: false,
    envEntries: [],
    headerEntries: [],
    originalSecretKeys: [],
  };
}

function createEditForm(server: McpServerView): ServerFormState {
  const toDrafts = (config: McpServerView['env_config']): EntryDraft[] =>
    Object.entries(config).map(([key, entry]) => (
      entry.kind === 'plain'
        ? {
            id: nextDraftId('entry'),
            key,
            kind: 'plain',
            value: entry.value,
            existingConfigured: false,
            secretAction: 'keep',
          }
        : {
            id: nextDraftId('entry'),
            key,
            kind: 'secret',
            value: '',
            existingConfigured: entry.configured,
            secretAction: 'keep',
          }
    ));

  const originalSecretKeys = Object.entries(
    server.transport === 'stdio' ? server.env_config : server.header_config,
  )
    .filter(([, entry]) => entry.kind === 'secret')
    .map(([key]) => key);

  return {
    serverId: server.server_id,
    name: server.name,
    transport: server.transport,
    command: server.command ?? '',
    args: server.args.map((value) => ({ id: nextDraftId('arg'), value })),
    cwd: server.cwd ?? '',
    hadCwd: server.cwd !== null && server.cwd !== '',
    url: server.url ?? '',
    allowPrivateNetwork: server.allow_private_network,
    envEntries: toDrafts(server.env_config),
    headerEntries: toDrafts(server.header_config),
    originalSecretKeys,
  };
}

function buildSaveRequest(form: ServerFormState): { ok: true; request: SaveMcpServerRequest } | { ok: false; error: string } {
  const name = form.name.trim();
  if (!name) return { ok: false, error: '请输入服务名称。' };

  const activeEntries = form.transport === 'stdio' ? form.envEntries : form.headerEntries;
  const configMap: Record<string, McpConfigEntryRequest> = {};
  const secretValues: Record<string, string> = {};
  const removeSecretKeys = new Set<string>();
  const currentSecretKeys = new Set<string>();

  for (const row of activeEntries) {
    const key = row.key.trim();
    if (!key) {
      if (row.value.trim()) {
        return { ok: false, error: `存在未填写键名的${form.transport === 'stdio' ? '环境变量' : '请求头'}项。` };
      }
      continue;
    }
    if (key in configMap) {
      return { ok: false, error: `键名「${key}」重复，请合并或删除重复项。` };
    }
    if (row.kind === 'plain') {
      configMap[key] = { kind: 'plain', value: row.value };
      if (form.originalSecretKeys.includes(key)) {
        removeSecretKeys.add(key);
      }
    } else {
      currentSecretKeys.add(key);
      configMap[key] = { kind: 'secret', secret_key: key };
      if (row.secretAction === 'delete') {
        removeSecretKeys.add(key);
      } else if (row.value.length > 0) {
        secretValues[key] = row.value;
      } else if (row.secretAction === 'update' && !row.existingConfigured) {
        return { ok: false, error: `凭据「${key}」尚未配置，请输入密钥值或移除该项。` };
      }
    }
  }

  for (const oldKey of form.originalSecretKeys) {
    if (!currentSecretKeys.has(oldKey)) {
      removeSecretKeys.add(oldKey);
    }
  }

  if (form.transport === 'stdio') {
    const command = form.command.trim();
    if (!command) return { ok: false, error: '请输入 stdio 可执行程序命令或绝对路径。' };
    const cwd = form.cwd.trim();
    return {
      ok: true,
      request: {
        server_id: form.serverId,
        name,
        transport: 'stdio',
        command,
        args: form.args.map((a) => a.value).filter((v) => v.length > 0),
        cwd: cwd ? cwd : null,
        clear_cwd: form.serverId !== null && form.hadCwd && !cwd,
        url: null,
        allow_private_network: false,
        env_config: configMap,
        header_config: {},
        secret_values: secretValues,
        remove_secret_keys: Array.from(removeSecretKeys),
      },
    };
  }

  const url = form.url.trim();
  if (!url) return { ok: false, error: '请输入 Streamable HTTP 服务地址（URL）。' };
  return {
    ok: true,
    request: {
      server_id: form.serverId,
      name,
      transport: 'streamable_http',
      command: null,
      args: [],
      cwd: null,
      clear_cwd: false,
      url,
      allow_private_network: form.allowPrivateNetwork,
      env_config: {},
      header_config: configMap,
      secret_values: secretValues,
      remove_secret_keys: Array.from(removeSecretKeys),
    },
  };
}

function summarizeFromServers(servers: McpServerView[], maxToolsPerRun = 32): McpSummary {
  const connected = servers.filter((s) => s.runtime.state === 'connected').length;
  const errored = servers.filter((s) => s.runtime.state === 'error').length;
  const enabled = servers.filter((s) => s.enabled).length;
  const effective = servers.reduce((acc, s) => acc + s.effective_allowed_tool_count, 0);
  return {
    total_servers: servers.length,
    enabled_servers: enabled,
    connected_servers: connected,
    error_servers: errored,
    effective_allowed_tools: effective,
    max_tools_per_run: maxToolsPerRun,
    exceeds_tool_limit: effective > maxToolsPerRun,
  };
}

export default function McpPanel({
  client,
  active,
  onSummaryChange,
}: {
  client?: ApiClient;
  active: boolean;
  onSummaryChange?: (summary: McpSummary) => void;
}) {
  const [servers, setServers] = useState<McpServerView[]>();
  const [summary, setSummary] = useState<McpSummary>();
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [notice, setNotice] = useState('');
  const [actionError, setActionError] = useState('');

  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const [expandedDiagnostics, setExpandedDiagnostics] = useState<Record<string, boolean>>({});
  const [expandedSchemas, setExpandedSchemas] = useState<Record<string, boolean>>({});
  const [toolsByServer, setToolsByServer] = useState<Record<string, McpToolView[]>>({});
  const [toolsLoading, setToolsLoading] = useState<Record<string, boolean>>({});
  const [toolsError, setToolsError] = useState<Record<string, string>>({});

  const [busyServers, setBusyServers] = useState<Record<string, string>>({});
  const [connectingAll, setConnectingAll] = useState(false);
  const [refreshingServers, setRefreshingServers] = useState<Record<string, boolean>>({});
  const [busyTools, setBusyTools] = useState<Record<string, boolean>>({});
  const [confirmDelete, setConfirmDelete] = useState<McpServerView>();

  const [formModal, setFormModal] = useState<ServerFormState>();
  const [formError, setFormError] = useState('');
  const [formSaving, setFormSaving] = useState(false);

  const [importOpen, setImportOpen] = useState(false);
  const [importRaw, setImportRaw] = useState('');
  const [importError, setImportError] = useState('');
  const [importSaving, setImportSaving] = useState(false);

  const fetchSeqRef = useRef(0);
  const toolSeqRef = useRef<Record<string, number>>({});
  const expandedToolsRef = useRef(expandedTools);
  expandedToolsRef.current = expandedTools;

  const fetchTools = useCallback(async (api: ApiClient, serverId: string, signal?: AbortSignal) => {
    const seq = (toolSeqRef.current[serverId] ?? 0) + 1;
    toolSeqRef.current[serverId] = seq;
    setToolsLoading((prev) => ({ ...prev, [serverId]: true }));
    setToolsError((prev) => ({ ...prev, [serverId]: '' }));
    try {
      const list = await api.listMcpServerTools(serverId, signal ?? new AbortController().signal);
      if (signal?.aborted || toolSeqRef.current[serverId] !== seq) return;
      setToolsByServer((prev) => ({ ...prev, [serverId]: list }));
    } catch (error) {
      if (signal?.aborted || toolSeqRef.current[serverId] !== seq) return;
      setToolsError((prev) => ({ ...prev, [serverId]: mcpErrorMessage(error, '无法加载工具列表。') }));
    } finally {
      if (!signal?.aborted && toolSeqRef.current[serverId] === seq) {
        setToolsLoading((prev) => ({ ...prev, [serverId]: false }));
      }
    }
  }, []);

  const fetchAuthoritativeState = useCallback(async (api: ApiClient, signal: AbortSignal, silent = false) => {
    const seq = ++fetchSeqRef.current;
    if (!silent) {
      setLoading(true);
      setLoadError('');
    }
    try {
      const [serverList, summaryData] = await Promise.all([
        api.listMcpServers(signal),
        api.getMcpSummary(signal).catch(() => undefined),
      ]);
      if (signal.aborted || fetchSeqRef.current !== seq) return;
      const resolvedSummary = summaryData ?? summarizeFromServers(serverList);
      setServers(serverList);
      setSummary(resolvedSummary);
      setLoadError('');
      onSummaryChange?.(resolvedSummary);

      // Refresh any currently expanded tool lists so live catalog/definition changes show up promptly.
      const expandedIds = Object.entries(expandedToolsRef.current)
        .filter(([, isOpen]) => isOpen)
        .map(([id]) => id)
        .filter((id) => serverList.some((s) => s.server_id === id));
      for (const sid of expandedIds) {
        void fetchTools(api, sid, signal);
      }
    } catch (error) {
      if (signal.aborted || fetchSeqRef.current !== seq) return;
      setLoadError(mcpErrorMessage(error, '无法加载 MCP 服务状态，请检查后端连接后重试。'));
    } finally {
      if (!signal.aborted && fetchSeqRef.current === seq && !silent) {
        setLoading(false);
      }
    }
  }, [fetchTools, onSummaryChange]);

  useEffect(() => {
    if (!client || !active) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll(silent: boolean) {
      await fetchAuthoritativeState(client!, controller.signal, silent);
      if (!controller.signal.aborted) {
        timer = setTimeout(() => void poll(true), MCP_POLL_INTERVAL_MS);
      }
    }

    void poll(servers !== undefined);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [active, client, fetchAuthoritativeState]);

  const effectiveSummary = useMemo(() => {
    if (summary) return summary;
    if (servers) return summarizeFromServers(servers);
    return {
      total_servers: 0,
      enabled_servers: 0,
      connected_servers: 0,
      error_servers: 0,
      effective_allowed_tools: 0,
      max_tools_per_run: 32,
      exceeds_tool_limit: false,
    };
  }, [servers, summary]);

  const importPreview = useMemo(
    () => (importRaw.trim() ? previewImportJson(importRaw) : undefined),
    [importRaw],
  );

  async function runServerAction(
    serverId: string,
    label: string,
    action: () => Promise<void>,
    fallbackError: string,
  ) {
    if (!client || busyServers[serverId]) return;
    setNotice('');
    setActionError('');
    setBusyServers((prev) => ({ ...prev, [serverId]: label }));
    try {
      await action();
      await fetchAuthoritativeState(client, new AbortController().signal, true);
    } catch (error) {
      setActionError(mcpErrorMessage(error, fallbackError));
      await fetchAuthoritativeState(client, new AbortController().signal, true);
    } finally {
      setBusyServers((prev) => {
        const next = { ...prev };
        delete next[serverId];
        return next;
      });
    }
  }

  async function handleToggleTools(serverId: string) {
    const nextOpen = !expandedTools[serverId];
    setExpandedTools((prev) => ({ ...prev, [serverId]: nextOpen }));
    if (nextOpen && client) {
      await fetchTools(client, serverId);
    }
  }

  async function handleTrust(server: McpServerView) {
    await runServerAction(
      server.server_id,
      '等待原生信任确认…',
      async () => {
        await mcpTrustServer(server.server_id, true);
        setNotice(`已确认信任并启用服务「${server.name}」。`);
      },
      `无法完成服务「${server.name}」的信任确认。`,
    );
  }

  async function handleRevokeTrust(server: McpServerView) {
    await runServerAction(
      server.server_id,
      '正在撤销信任…',
      async () => {
        await mcpRevokeTrust(server.server_id);
        setNotice(`已撤销服务「${server.name}」的信任并断开连接，所有工具授权已重置。`);
      },
      `无法撤销服务「${server.name}」的信任。`,
    );
  }

  async function handleToggleEnabled(server: McpServerView) {
    const nextEnabled = !server.enabled;
    await runServerAction(
      server.server_id,
      nextEnabled ? '正在启用…' : '正在停用…',
      async () => {
        await mcpSetServerEnabled(server.server_id, nextEnabled);
        setNotice(nextEnabled ? `已启用服务「${server.name}」。` : `已停用服务「${server.name}」并断开连接。`);
      },
      nextEnabled ? `无法启用服务「${server.name}」。` : `无法停用服务「${server.name}」。`,
    );
  }

  async function handleConnect(server: McpServerView) {
    await runServerAction(
      server.server_id,
      server.runtime.state === 'connected' ? '正在重连…' : '正在连接…',
      async () => {
        await mcpConnectServer(server.server_id);
        setNotice(`已连接服务「${server.name}」。`);
        setExpandedTools((prev) => ({ ...prev, [server.server_id]: true }));
        if (client) await fetchTools(client, server.server_id);
      },
      `连接服务「${server.name}」失败。`,
    );
  }

  const connectableServers = useMemo(
    () => (servers ?? []).filter((server) => (
      server.enabled
      && server.trust_valid
      && server.credentials_complete
      && server.runtime.state !== 'connected'
      && server.runtime.state !== 'connecting'
      && server.runtime.state !== 'disconnecting'
      && !busyServers[server.server_id]
    )),
    [servers, busyServers],
  );

  async function handleConnectAll() {
    if (!client || connectingAll || connectableServers.length === 0) return;
    setActionError('');
    setNotice('');
    setConnectingAll(true);
    const targets = connectableServers;
    setBusyServers((prev) => {
      const next = { ...prev };
      for (const srv of targets) next[srv.server_id] = '正在连接…';
      return next;
    });
    const results = await Promise.allSettled(
      targets.map((srv) => mcpConnectServer(srv.server_id)),
    );
    const succeeded: string[] = [];
    const failed: string[] = [];
    results.forEach((result, idx) => {
      if (result.status === 'fulfilled') {
        succeeded.push(targets[idx].name);
      } else {
        failed.push(targets[idx].name);
      }
    });
    setBusyServers((prev) => {
      const next = { ...prev };
      for (const srv of targets) delete next[srv.server_id];
      return next;
    });
    setConnectingAll(false);
    await fetchAuthoritativeState(client, new AbortController().signal, false);
    if (failed.length === 0) {
      setNotice(`已连接全部 ${succeeded.length} 个服务。`);
    } else if (succeeded.length > 0) {
      setNotice(`已连接 ${succeeded.length} 个服务。`);
      setActionError(`以下服务连接失败：${failed.join('、')}。`);
    } else {
      setActionError(`全部 ${failed.length} 个服务连接失败，请展开诊断查看详情。`);
    }
  }

  async function handleDisconnect(server: McpServerView) {
    await runServerAction(
      server.server_id,
      '正在断开…',
      async () => {
        await mcpDisconnectServer(server.server_id);
        setNotice(`已断开服务「${server.name}」。`);
      },
      `断开服务「${server.name}」失败。`,
    );
  }

  async function handleRefreshTools(server: McpServerView) {
    if (!client || busyServers[server.server_id]) return;
    setRefreshingServers((prev) => ({ ...prev, [server.server_id]: true }));
    try {
      await runServerAction(
        server.server_id,
        '正在刷新工具…',
        async () => {
          const updatedTools = await mcpRefreshTools(server.server_id);
          setToolsByServer((prev) => ({ ...prev, [server.server_id]: updatedTools }));
          setExpandedTools((prev) => ({ ...prev, [server.server_id]: true }));
          setNotice(`已刷新服务「${server.name}」的工具列表（共 ${updatedTools.length} 个）。`);
        },
        `刷新服务「${server.name}」工具列表失败。`,
      );
    } finally {
      setRefreshingServers((prev) => {
        const next = { ...prev };
        delete next[server.server_id];
        return next;
      });
    }
  }

  async function handleDelete(server: McpServerView) {
    await runServerAction(
      server.server_id,
      '正在删除…',
      async () => {
        await mcpDeleteServer(server.server_id);
        setConfirmDelete(undefined);
        setToolsByServer((prev) => {
          const next = { ...prev };
          delete next[server.server_id];
          return next;
        });
        setNotice(`已删除 MCP 服务「${server.name}」。`);
      },
      `删除服务「${server.name}」失败。`,
    );
  }

  async function handleToggleToolPermission(server: McpServerView, tool: McpToolView) {
    if (!client || busyTools[tool.tool_id]) return;
    const approving = tool.permission_state !== 'allowed' || tool.definition_changed;
    setNotice('');
    setActionError('');
    setBusyTools((prev) => ({ ...prev, [tool.tool_id]: true }));
    try {
      const updated = await mcpSetToolPermission({
        server_id: server.server_id,
        tool_id: tool.tool_id,
        permission_state: approving ? 'allowed' : 'denied',
        expected_definition_hash: approving ? tool.definition_hash : null,
        expected_config_revision: approving ? server.config_revision : null,
      });
      setToolsByServer((prev) => ({
        ...prev,
        [server.server_id]: (prev[server.server_id] ?? []).map((item) =>
          item.tool_id === updated.tool_id ? updated : item),
      }));
      setNotice(
        approving
          ? `已授权工具「${tool.original_tool_name}」。`
          : `已撤销工具「${tool.original_tool_name}」的授权。`,
      );
      await fetchAuthoritativeState(client, new AbortController().signal, true);
    } catch (error) {
      setActionError(mcpErrorMessage(error, `更新工具「${tool.original_tool_name}」授权失败。`));
      // Always re-fetch authoritative state on failure; never keep optimistic state.
      await Promise.all([
        fetchTools(client, server.server_id),
        fetchAuthoritativeState(client, new AbortController().signal, true),
      ]);
    } finally {
      setBusyTools((prev) => {
        const next = { ...prev };
        delete next[tool.tool_id];
        return next;
      });
    }
  }

  async function submitServerForm(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!client || !formModal || formSaving) return;
    setFormError('');
    const built = buildSaveRequest(formModal);
    if (!built.ok) {
      setFormError(built.error);
      return;
    }
    setFormSaving(true);
    setNotice('');
    setActionError('');
    try {
      const saved = await mcpSaveServer(built.request);
      setFormModal(undefined);
      setNotice(
        formModal.serverId
          ? `已保存服务「${saved.name}」的配置。若修改了关键配置，需重新确认信任。`
          : `已添加服务草稿「${saved.name}」（默认停用且未信任），请确认信任后再连接。`,
      );
      await fetchAuthoritativeState(client, new AbortController().signal, true);
    } catch (error) {
      setFormError(mcpErrorMessage(error, '保存 MCP 服务配置失败，请检查输入后重试。'));
    } finally {
      setFormSaving(false);
    }
  }

  async function submitImportJson(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!client || importSaving) return;
    const preview = previewImportJson(importRaw);
    if (!preview.ok) {
      setImportError(preview.error);
      return;
    }
    setImportSaving(true);
    setImportError('');
    setNotice('');
    setActionError('');
    try {
      const result = await mcpImportServersJson(importRaw);
      // Immediately wipe raw JSON (which may contain plaintext keys) from component memory.
      setImportRaw('');
      setImportOpen(false);
      const warnSuffix = result.warnings.length > 0 ? ` ${result.warnings.join(' ')}` : '';
      setNotice(`已导入 ${result.servers.length} 个未信任草稿服务。${warnSuffix}`);
      await fetchAuthoritativeState(client, new AbortController().signal, true);
    } catch (error) {
      setImportError(mcpErrorMessage(error, '导入 JSON 失败，请检查配置格式或服务名称是否冲突。'));
    } finally {
      setImportSaving(false);
    }
  }

  function updateEntryList(
    transport: McpTransportKind,
    updater: (entries: EntryDraft[]) => EntryDraft[],
  ) {
    setFormModal((prev) => {
      if (!prev) return prev;
      if (transport === 'stdio') {
        return { ...prev, envEntries: updater(prev.envEntries) };
      }
      return { ...prev, headerEntries: updater(prev.headerEntries) };
    });
  }

  const activeEntries = formModal
    ? (formModal.transport === 'stdio' ? formModal.envEntries : formModal.headerEntries)
    : [];

  return (
    <section className="mcp-panel" aria-label="MCP 服务管理">
      <header className="mcp-header">
        <div className="mcp-header-title">
          <h2>MCP 服务</h2>
          <p>管理外部 Model Context Protocol 工具服务。所有新增或导入服务默认停用且未信任，需经 macOS 原生弹窗确认与逐工具授权后方可调用。</p>
        </div>
        <div className="mcp-header-actions">
          <button
            type="button"
            className="mcp-btn primary"
            disabled={!client}
            onClick={() => {
              setFormError('');
              setFormModal(createEmptyForm());
            }}>
            + 添加服务
          </button>
          <button
            type="button"
            className="mcp-btn secondary"
            disabled={!client}
            onClick={() => {
              setImportError('');
              setImportRaw('');
              setImportOpen(true);
            }}>
            导入 JSON
          </button>
          {connectableServers.length > 0 && (
            <button
              type="button"
              className="mcp-btn secondary"
              disabled={!client || connectingAll}
              onClick={() => void handleConnectAll()}>
              {connectingAll ? '正在连接全部…' : `全部连接 (${connectableServers.length})`}
            </button>
          )}
          <button
            type="button"
            className="mcp-btn ghost"
            disabled={!client || loading}
            onClick={() => client && void fetchAuthoritativeState(client, new AbortController().signal, false)}>
            刷新状态
          </button>
        </div>
      </header>

      <div className="mcp-summary-grid" aria-label="MCP 状态摘要">
        <div className="mcp-stat-card">
          <span className="mcp-stat-label">已配置服务</span>
          <strong className="mcp-stat-value">{effectiveSummary.total_servers}</strong>
        </div>
        <div className="mcp-stat-card">
          <span className="mcp-stat-label">已连接服务</span>
          <strong className="mcp-stat-value">{effectiveSummary.connected_servers}</strong>
        </div>
        <div className={`mcp-stat-card ${effectiveSummary.exceeds_tool_limit ? 'is-danger' : ''}`}>
          <span className="mcp-stat-label">当前有效授权工具</span>
          <strong className="mcp-stat-value">
            {effectiveSummary.effective_allowed_tools} / {effectiveSummary.max_tools_per_run}
          </strong>
        </div>
      </div>

      {effectiveSummary.exceeds_tool_limit && (
        <div role="alert" className="mcp-banner danger">
          当前有效授权工具总数（{effectiveSummary.effective_allowed_tools}）已超过单轮对话上限（{effectiveSummary.max_tools_per_run}）。请取消部分工具授权后再在聊天中开启 MCP。
        </div>
      )}

      {notice && <p role="status" className="mcp-banner success">{notice}</p>}
      {actionError && <p role="alert" className="mcp-banner danger">{actionError}</p>}
      {loadError && (
        <div role="alert" className="mcp-banner danger">
          <span>{loadError}</span>{' '}
          <button
            type="button"
            className="mcp-link-btn"
            onClick={() => client && void fetchAuthoritativeState(client, new AbortController().signal, false)}>
            重试加载
          </button>
        </div>
      )}

      {loading && servers === undefined && <p role="status" className="mcp-loading">正在读取 MCP 服务列表…</p>}

      {servers && servers.length === 0 && !loadError && (
        <div className="mcp-empty-state">
          <div className="mcp-empty-icon" aria-hidden="true">🔌</div>
          <h3>尚未配置任何 MCP 服务</h3>
          <p>
            您可以手动添加本地 <code>stdio</code> 服务或远程 <code>Streamable HTTP</code> 服务，也可以粘贴标准的 <code>mcpServers</code> JSON 草稿导入。
          </p>
          <div className="mcp-empty-actions">
            <button
              type="button"
              className="mcp-btn primary"
              onClick={() => {
                setFormError('');
                setFormModal(createEmptyForm());
              }}>
              + 添加服务
            </button>
            <button
              type="button"
              className="mcp-btn secondary"
              onClick={() => {
                setImportError('');
                setImportRaw('');
                setImportOpen(true);
              }}>
              导入 JSON
            </button>
          </div>
        </div>
      )}

      {servers && servers.length > 0 && (
        <div className="mcp-server-list">
          {servers.map((server) => {
            const busyLabel = busyServers[server.server_id];
            const isBusy = Boolean(busyLabel);
            const isRefreshing = Boolean(refreshingServers[server.server_id]) || server.runtime.catalog_stale;
            const trustBadge = trustBadgeInfo(server);
            const isConnected = server.runtime.state === 'connected';
            const isConnecting = server.runtime.state === 'connecting' || server.runtime.state === 'disconnecting';
            const canConnect = server.enabled && server.trust_valid && server.credentials_complete && !isBusy && !isConnecting;
            const secretCount = Object.values(
              server.transport === 'stdio' ? server.env_config : server.header_config,
            ).filter((entry) => entry.kind === 'secret').length;
            const toolsOpen = Boolean(expandedTools[server.server_id]);
            const diagOpen = Boolean(expandedDiagnostics[server.server_id]);
            const serverTools = toolsByServer[server.server_id];

            return (
              <article key={server.server_id} className="mcp-server-card" data-server-id={server.server_id}>
                <div className="mcp-card-top">
                  <div className="mcp-card-title-row">
                    <h3>{server.name}</h3>
                    <span className="badge badge-transport">
                      {server.transport === 'stdio' ? 'stdio' : 'Streamable HTTP'}
                    </span>
                    <span className={`mcp-pill ${server.enabled ? 'tone-info' : 'tone-muted'}`}>
                      {server.enabled ? '已启用' : '已停用'}
                    </span>
                    <span className={`mcp-pill tone-${trustBadge.tone}`}>
                      {trustBadge.label}
                    </span>
                    <span className={`mcp-pill runtime-${server.runtime.state}`}>
                      {runtimeLabel(server.runtime.state)}
                    </span>
                    <span className={`mcp-pill ${server.credentials_complete ? (secretCount > 0 ? 'tone-ok' : 'tone-muted') : 'tone-danger'}`}>
                      {!server.credentials_complete
                        ? `缺少凭据（${server.missing_secret_keys.length}）`
                        : secretCount > 0
                          ? `凭据已配置（${secretCount}）`
                          : '无需凭据'}
                    </span>
                  </div>

                  <div className="mcp-card-target">
                    {server.transport === 'stdio' ? (
                      <code>{server.resolved_command || server.command} {server.args.join(' ')}</code>
                    ) : (
                      <code>{server.url}</code>
                    )}
                  </div>

                  <div className="mcp-card-metrics">
                    <span>已发现工具：{server.discovered_tool_count}</span>
                    <span>历史已授权：{server.allowed_tool_count}</span>
                    <strong>当前有效授权：{isRefreshing ? 0 : server.effective_allowed_tool_count}</strong>
                    {server.runtime.tools_truncated && (
                      <span className="mcp-inline-warn">工具目录超出上限已截断</span>
                    )}
                    {isRefreshing && (
                      <span className="mcp-inline-warn">目录刷新中 · 旧工具不可调用</span>
                    )}
                  </div>
                </div>

                {server.high_risk_command && (
                  <p className="mcp-card-notice warn">
                    高风险命令提示：{server.high_risk_reasons.join('；') || '检测到动态执行器或脚本解释器，请谨慎核对命令来源。'}
                  </p>
                )}

                {server.transport === 'stdio' && server.executable_exists === false && (
                  <p className="mcp-card-notice danger">
                    未找到可执行程序：{server.command}（配置已保留，请编辑确认路径）。
                  </p>
                )}

                {!server.credentials_complete && (
                  <p className="mcp-card-notice danger">
                    缺少钥匙串凭据：{server.missing_secret_keys.join('、')}。请点击「编辑」补充凭据后再连接。
                  </p>
                )}

                {(server.runtime.last_error_code || server.runtime.last_error_message) && (
                  <div role="alert" className="mcp-card-error">
                    <strong>最近错误（已脱敏）：</strong>
                    <span>{server.runtime.last_error_message || mcpErrorMessage(server.runtime.last_error_code, '连接失败。')}</span>
                    {server.runtime.last_error_code && <code>{server.runtime.last_error_code}</code>}
                  </div>
                )}

                {busyLabel && <p role="status" className="mcp-busy-hint">{busyLabel}</p>}

                <div className="mcp-card-actions">
                  {!server.trust_valid ? (
                    <button
                      type="button"
                      className="mcp-btn primary sm"
                      disabled={isBusy}
                      onClick={() => void handleTrust(server)}>
                      {server.trust_state === 'trusted' ? '重新信任' : '信任确认'}
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="mcp-btn ghost sm"
                      disabled={isBusy}
                      onClick={() => void handleRevokeTrust(server)}>
                      撤销信任
                    </button>
                  )}

                  <button
                    type="button"
                    className="mcp-btn secondary sm"
                    disabled={isBusy || (!server.enabled && !server.trust_valid)}
                    title={!server.enabled && !server.trust_valid ? '请先完成信任确认再启用服务' : undefined}
                    onClick={() => void handleToggleEnabled(server)}>
                    {server.enabled ? '停用' : '启用'}
                  </button>

                  {isConnected ? (
                    <>
                      <button
                        type="button"
                        className="mcp-btn secondary sm"
                        disabled={isBusy}
                        onClick={() => void handleDisconnect(server)}>
                        断开
                      </button>
                      <button
                        type="button"
                        className="mcp-btn secondary sm"
                        disabled={!canConnect}
                        onClick={() => void handleConnect(server)}>
                        重连
                      </button>
                      <button
                        type="button"
                        className="mcp-btn secondary sm"
                        disabled={isBusy || isRefreshing}
                        onClick={() => void handleRefreshTools(server)}>
                        刷新工具
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      className="mcp-btn secondary sm"
                      disabled={!canConnect}
                      title={
                        !server.trust_valid
                          ? '请先完成信任确认'
                          : !server.enabled
                            ? '请先启用服务'
                            : !server.credentials_complete
                              ? '请先补充必需的钥匙串凭据'
                              : undefined
                      }
                      onClick={() => void handleConnect(server)}>
                      {server.runtime.state === 'error' ? '重连' : '连接'}
                    </button>
                  )}

                  <button
                    type="button"
                    className="mcp-btn ghost sm"
                    aria-expanded={toolsOpen}
                    onClick={() => void handleToggleTools(server.server_id)}>
                    {toolsOpen ? '收起工具列表' : `展开工具列表（${server.discovered_tool_count}）`}
                  </button>

                  <button
                    type="button"
                    className="mcp-btn ghost sm"
                    aria-expanded={diagOpen}
                    onClick={() => setExpandedDiagnostics((prev) => ({ ...prev, [server.server_id]: !diagOpen }))}>
                    {diagOpen ? '收起诊断' : '诊断信息'}
                  </button>

                  <button
                    type="button"
                    className="mcp-btn ghost sm"
                    disabled={isBusy}
                    onClick={() => {
                      setFormError('');
                      setFormModal(createEditForm(server));
                    }}>
                    编辑
                  </button>

                  <button
                    type="button"
                    className="mcp-btn danger sm"
                    disabled={isBusy}
                    onClick={() => setConfirmDelete(server)}>
                    删除
                  </button>
                </div>

                {confirmDelete?.server_id === server.server_id && (
                  <div className="mcp-delete-confirm" role="alertdialog" aria-label={`确认删除 ${server.name}`}>
                    <span>确认删除服务「{server.name}」并清理关联钥匙串凭据？（保留历史聊天工具审计记录）</span>
                    <div className="mcp-delete-confirm-actions">
                      <button
                        type="button"
                        className="mcp-btn danger sm"
                        disabled={isBusy}
                        onClick={() => void handleDelete(server)}>
                        确认删除
                      </button>
                      <button
                        type="button"
                        className="mcp-btn ghost sm"
                        disabled={isBusy}
                        onClick={() => setConfirmDelete(undefined)}>
                        取消
                      </button>
                    </div>
                  </div>
                )}

                {diagOpen && (
                  <div className="mcp-diagnostics-box" aria-label={`${server.name} 脱敏诊断信息`}>
                    <dl className="mcp-diag-grid">
                      <div>
                        <dt>配置版本 / 指纹</dt>
                        <dd><code>r{server.config_revision} · {server.current_fingerprint.slice(0, 12)}…</code></dd>
                      </div>
                      {server.transport === 'stdio' ? (
                        <>
                          <div>
                            <dt>解析后可执行路径</dt>
                            <dd><code>{server.resolved_command || '未找到可执行文件'}</code></dd>
                          </div>
                          <div>
                            <dt>工作目录</dt>
                            <dd><code>{server.cwd || '（默认）'}</code></dd>
                          </div>
                        </>
                      ) : (
                        <div>
                          <dt>私有网络策略</dt>
                          <dd>
                            {server.allow_private_network ? '已显式允许私网' : '默认禁止私网'}
                            {server.private_network_target ? ' · 目标属于私有地址' : ''}
                          </dd>
                        </div>
                      )}
                      <div>
                        <dt>协议 / 上报服务</dt>
                        <dd>
                          {server.runtime.protocol_version || '—'}
                          {server.runtime.server_name_reported
                            ? ` · ${server.runtime.server_name_reported}${server.runtime.server_version_reported ? ` v${server.runtime.server_version_reported}` : ''}`
                            : ''}
                        </dd>
                      </div>
                      <div>
                        <dt>连接代际 / 目录版本</dt>
                        <dd>gen {server.runtime.generation} · catalog v{server.runtime.catalog_version}</dd>
                      </div>
                    </dl>
                    {server.runtime.stderr_tail.length > 0 && (
                      <div className="mcp-stderr-block">
                        <span className="mcp-stderr-label">脱敏 stderr 尾部：</span>
                        <pre>{server.runtime.stderr_tail.join('\n')}</pre>
                      </div>
                    )}
                  </div>
                )}

                {toolsOpen && (
                  <div className="mcp-tools-section" aria-label={`${server.name} 工具列表`}>
                    {isRefreshing && (
                      <p role="status" className="mcp-card-notice warn">
                        工具目录正在刷新或已过期，刷新完成前旧工具不可调用。
                      </p>
                    )}
                    {toolsLoading[server.server_id] && !serverTools && (
                      <p role="status" className="mcp-loading-inline">正在加载工具列表…</p>
                    )}
                    {toolsError[server.server_id] && (
                      <p role="alert" className="mcp-card-notice danger">{toolsError[server.server_id]}</p>
                    )}
                    {serverTools && serverTools.length === 0 && (
                      <p className="mcp-tools-empty">
                        {isConnected ? '该服务未暴露任何工具。' : '尚未发现工具，请先完成信任并连接服务。'}
                      </p>
                    )}
                    {serverTools && serverTools.length > 0 && (
                      <ul className="mcp-tool-list">
                        {serverTools.map((tool) => {
                          const readOnlyHint = tool.annotations?.readOnlyHint === true;
                          const openWorldHint = tool.annotations?.openWorldHint === true;
                          const blockedReason = !tool.present_in_last_discovery
                            ? '该工具已从最新发现中移除，不可授权。'
                            : !tool.schema_compatible
                              ? `Schema 不兼容：${tool.incompatibility_reason || '不符合安全规范'}`
                              : tool.is_destructive
                                ? '后端判定为破坏性工具，当前版本禁止授权。'
                                : undefined;
                          const canAuthorize = !blockedReason && !isRefreshing && !busyTools[tool.tool_id];
                          const isApproved = tool.permission_state === 'allowed' && !tool.definition_changed;
                          const isEffective = !isRefreshing && tool.effective_allowed;
                          const schemaOpen = Boolean(expandedSchemas[tool.tool_id]);

                          return (
                            <li key={tool.tool_id} className="mcp-tool-item">
                              <div className="mcp-tool-main">
                                <div className="mcp-tool-head">
                                  <code className="mcp-tool-name">{tool.original_tool_name}</code>
                                  {tool.title && <span className="mcp-tool-title">{tool.title}</span>}

                                  <span className={`mcp-pill ${tool.schema_compatible ? 'tone-ok' : 'tone-danger'}`}>
                                    {tool.schema_compatible ? 'Schema 兼容' : 'Schema 不兼容'}
                                  </span>

                                  {tool.is_destructive && (
                                    <span className="mcp-pill tone-danger">破坏性工具 · 禁止授权</span>
                                  )}
                                  {readOnlyHint && (
                                    <span className="mcp-pill tone-muted" title="由外部服务声明，仅供参考，不会自动授权">
                                      服务端标注只读（仅供参考）
                                    </span>
                                  )}
                                  {openWorldHint && (
                                    <span className="mcp-pill tone-warn">访问外部网络（参考）</span>
                                  )}
                                  {!tool.present_in_last_discovery && (
                                    <span className="mcp-pill tone-danger">已失效 / 不存在</span>
                                  )}
                                  {tool.definition_changed && (
                                    <span className="mcp-pill tone-warn">定义已变更 · 需重新授权</span>
                                  )}

                                  <span className={`mcp-pill ${isApproved ? 'tone-info' : 'tone-muted'}`}>
                                    用户授权：{isApproved ? '已授权' : '未授权'}
                                  </span>
                                  <span className={`mcp-pill ${isEffective ? 'tone-ok' : 'tone-muted'}`}>
                                    生效状态：{isRefreshing ? '刷新中不可用' : isEffective ? '当前有效' : '当前未生效'}
                                  </span>
                                </div>

                                {tool.description && <p className="mcp-tool-desc">{tool.description}</p>}
                                {blockedReason && <p className="mcp-tool-block-note">{blockedReason}</p>}

                                <div className="mcp-tool-footer">
                                  <button
                                    type="button"
                                    className="mcp-link-btn"
                                    aria-expanded={schemaOpen}
                                    onClick={() => setExpandedSchemas((prev) => ({ ...prev, [tool.tool_id]: !schemaOpen }))}>
                                    {schemaOpen ? '收起参数 Schema' : '查看参数 Schema'}
                                  </button>
                                </div>

                                {schemaOpen && (
                                  <pre className="mcp-schema-pre">
                                    {JSON.stringify(tool.input_schema, null, 2)}
                                  </pre>
                                )}
                              </div>

                              <div className="mcp-tool-control">
                                <button
                                  type="button"
                                  className={`mcp-btn sm ${isApproved ? 'ghost' : 'primary'}`}
                                  disabled={isApproved ? Boolean(busyTools[tool.tool_id]) : !canAuthorize}
                                  aria-label={
                                    isApproved
                                      ? `撤销授权 ${tool.original_tool_name}`
                                      : tool.definition_changed
                                        ? `重新授权 ${tool.original_tool_name}`
                                        : `授权 ${tool.original_tool_name}`
                                  }
                                  onClick={() => void handleToggleToolPermission(server, tool)}>
                                  {busyTools[tool.tool_id]
                                    ? '处理中…'
                                    : isApproved
                                      ? '撤销授权'
                                      : tool.definition_changed
                                        ? '重新授权'
                                        : '授权启用'}
                                </button>
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                )}
              </article>
            );
          })}
        </div>
      )}

      {formModal && (
        <div className="mcp-modal-overlay" role="dialog" aria-modal="true" aria-labelledby="mcp-form-title">
          <div className="mcp-modal-card">
            <div className="mcp-modal-header">
              <h3 id="mcp-form-title">{formModal.serverId ? '编辑 MCP 服务' : '添加 MCP 服务'}</h3>
              <button
                type="button"
                className="icon-close-btn"
                aria-label="关闭服务表单"
                disabled={formSaving}
                onClick={() => setFormModal(undefined)}>
                ✕
              </button>
            </div>

            <form className="mcp-form" onSubmit={submitServerForm}>
              <label className="mcp-field">
                <span>服务名称</span>
                <input
                  type="text"
                  value={formModal.name}
                  maxLength={100}
                  placeholder="例如：Local Stdio Test"
                  onChange={(e) => setFormModal((prev) => prev && { ...prev, name: e.target.value })}
                />
              </label>

              <fieldset className="mcp-transport-fieldset">
                <legend>传输方式（Transport）</legend>
                <div className="mcp-transport-options">
                  <label className={`mcp-radio-pill ${formModal.transport === 'stdio' ? 'active' : ''}`}>
                    <input
                      type="radio"
                      name="mcp-transport"
                      value="stdio"
                      checked={formModal.transport === 'stdio'}
                      onChange={() => setFormModal((prev) => prev && { ...prev, transport: 'stdio' })}
                    />
                    <span>stdio（本地外部程序）</span>
                  </label>
                  <label className={`mcp-radio-pill ${formModal.transport === 'streamable_http' ? 'active' : ''}`}>
                    <input
                      type="radio"
                      name="mcp-transport"
                      value="streamable_http"
                      checked={formModal.transport === 'streamable_http'}
                      onChange={() => setFormModal((prev) => prev && { ...prev, transport: 'streamable_http' })}
                    />
                    <span>Streamable HTTP（网络服务）</span>
                  </label>
                </div>
              </fieldset>

              {formModal.transport === 'stdio' ? (
                <>
                  <p className="mcp-security-note">
                    安全说明：本地 stdio 外部程序以当前系统用户权限运行，<strong>不具备操作系统级文件沙箱</strong>。新增或修改关键配置后，必须经 macOS 原生弹窗重新确认信任。
                  </p>

                  <label className="mcp-field">
                    <span>可执行命令（Command）</span>
                    <input
                      type="text"
                      value={formModal.command}
                      placeholder="推荐绝对路径，例如 /opt/homebrew/bin/python3"
                      onChange={(e) => setFormModal((prev) => prev && { ...prev, command: e.target.value })}
                    />
                  </label>

                  <div className="mcp-list-editor">
                    <div className="mcp-list-editor-head">
                      <span>启动参数（Args，逐项填写，保留含空格路径）</span>
                      <button
                        type="button"
                        className="mcp-link-btn"
                        onClick={() => setFormModal((prev) => prev && {
                          ...prev,
                          args: [...prev.args, { id: nextDraftId('arg'), value: '' }],
                        })}>
                        + 添加参数
                      </button>
                    </div>
                    {formModal.args.length === 0 ? (
                      <p className="mcp-hint-muted">暂无参数，可点击「+ 添加参数」逐项添加。</p>
                    ) : (
                      formModal.args.map((arg, index) => (
                        <div key={arg.id} className="mcp-arg-row">
                          <input
                            type="text"
                            aria-label={`启动参数 ${index + 1}`}
                            value={arg.value}
                            placeholder={`参数 #${index + 1}（可含空格，如 /path/with space/server.py）`}
                            onChange={(e) => setFormModal((prev) => prev && {
                              ...prev,
                              args: prev.args.map((item) => item.id === arg.id ? { ...item, value: e.target.value } : item),
                            })}
                          />
                          <button
                            type="button"
                            className="mcp-btn ghost sm"
                            aria-label={`移除参数 ${index + 1}`}
                            onClick={() => setFormModal((prev) => prev && {
                              ...prev,
                              args: prev.args.filter((item) => item.id !== arg.id),
                            })}>
                            移除
                          </button>
                        </div>
                      ))
                    )}
                  </div>

                  <label className="mcp-field">
                    <span>工作目录（可选 CWD）</span>
                    <input
                      type="text"
                      value={formModal.cwd}
                      placeholder="留空则使用默认工作目录"
                      onChange={(e) => setFormModal((prev) => prev && { ...prev, cwd: e.target.value })}
                    />
                  </label>
                </>
              ) : (
                <>
                  <p className="mcp-security-note">
                    安全说明：Streamable HTTP 非本地回环地址必须使用 <code>https://</code>，禁用 HTTP 重定向，且外部工具调用可能产生数据外发。
                  </p>

                  <label className="mcp-field">
                    <span>服务地址（Server URL）</span>
                    <input
                      type="url"
                      value={formModal.url}
                      placeholder="例如 https://mcp.example.com/mcp 或 http://127.0.0.1:8787/mcp"
                      onChange={(e) => setFormModal((prev) => prev && { ...prev, url: e.target.value })}
                    />
                  </label>

                  <label className="mcp-checkbox-field">
                    <input
                      type="checkbox"
                      checked={formModal.allowPrivateNetwork}
                      onChange={(e) => setFormModal((prev) => prev && {
                        ...prev,
                        allowPrivateNetwork: e.target.checked,
                      })}
                    />
                    <span>显式允许访问私有局域网地址（RFC1918 / 内网服务）</span>
                  </label>
                </>
              )}

              <div className="mcp-list-editor">
                <div className="mcp-list-editor-head">
                  <span>{formModal.transport === 'stdio' ? '环境变量与凭据' : 'HTTP 请求头与凭据'}</span>
                  <button
                    type="button"
                    className="mcp-link-btn"
                    onClick={() => updateEntryList(formModal.transport, (list) => [
                      ...list,
                      {
                        id: nextDraftId('entry'),
                        key: '',
                        kind: 'plain',
                        value: '',
                        existingConfigured: false,
                        secretAction: 'keep',
                      },
                    ])}>
                    + 添加一项
                  </button>
                </div>

                {activeEntries.length === 0 ? (
                  <p className="mcp-hint-muted">
                    暂无{formModal.transport === 'stdio' ? '环境变量' : '请求头'}。密钥项将隔离存入 macOS 钥匙串，不落 SQLite 也不回显明文。
                  </p>
                ) : (
                  activeEntries.map((entry, idx) => (
                    <div key={entry.id} className="mcp-kv-row">
                      <input
                        type="text"
                        aria-label={`配置键 ${idx + 1}`}
                        placeholder={formModal.transport === 'stdio' ? '变量名（如 API_KEY）' : 'Header 名（如 Authorization）'}
                        value={entry.key}
                        onChange={(e) => updateEntryList(formModal.transport, (list) =>
                          list.map((item) => item.id === entry.id ? { ...item, key: e.target.value } : item))}
                      />
                      <select
                        aria-label={`配置类型 ${idx + 1}`}
                        value={entry.kind}
                        onChange={(e) => {
                          const nextKind = e.target.value as 'plain' | 'secret';
                          updateEntryList(formModal.transport, (list) =>
                            list.map((item) => item.id === entry.id
                              ? { ...item, kind: nextKind, value: '', secretAction: nextKind === 'secret' ? 'keep' : 'keep' }
                              : item));
                        }}>
                        <option value="plain">明文</option>
                        <option value="secret">密钥（钥匙串）</option>
                      </select>

                      {entry.kind === 'plain' ? (
                        <input
                          type="text"
                          aria-label={`明文值 ${idx + 1}`}
                          placeholder="明文值"
                          value={entry.value}
                          onChange={(e) => updateEntryList(formModal.transport, (list) =>
                            list.map((item) => item.id === entry.id ? { ...item, value: e.target.value } : item))}
                        />
                      ) : (
                        <div className="mcp-secret-cell">
                          {entry.existingConfigured && (
                            <div className="mcp-secret-mode-bar">
                              <span className="mcp-secret-tag">
                                {entry.secretAction === 'delete' ? '将删除已存密钥' : '已配置（不回显）'}
                              </span>
                              <select
                                aria-label={`凭据操作 ${entry.key || idx + 1}`}
                                value={entry.secretAction}
                                onChange={(e) => {
                                  const action = e.target.value as 'keep' | 'update' | 'delete';
                                  updateEntryList(formModal.transport, (list) =>
                                    list.map((item) => item.id === entry.id
                                      ? { ...item, secretAction: action, value: action === 'update' ? item.value : '' }
                                      : item));
                                }}>
                                <option value="keep">留空保持原值</option>
                                <option value="update">更新凭据</option>
                                <option value="delete">删除凭据</option>
                              </select>
                            </div>
                          )}
                          {(!entry.existingConfigured || entry.secretAction === 'update') && (
                            <input
                              type="password"
                              autoComplete="new-password"
                              aria-label={`密钥值 ${entry.key || idx + 1}`}
                              placeholder={entry.existingConfigured ? '输入新密钥以替换原值' : '输入密钥（存入钥匙串）'}
                              value={entry.value}
                              onChange={(e) => updateEntryList(formModal.transport, (list) =>
                                list.map((item) => item.id === entry.id
                                  ? { ...item, value: e.target.value, secretAction: e.target.value ? 'update' : item.secretAction }
                                  : item))}
                            />
                          )}
                        </div>
                      )}

                      <button
                        type="button"
                        className="mcp-btn ghost sm"
                        aria-label={`删除配置项 ${entry.key || idx + 1}`}
                        onClick={() => updateEntryList(formModal.transport, (list) =>
                          list.filter((item) => item.id !== entry.id))}>
                        移除
                      </button>
                    </div>
                  ))
                )}
              </div>

              {formError && <p role="alert" className="mcp-banner danger">{formError}</p>}

              <div className="mcp-modal-actions">
                <button
                  type="button"
                  className="mcp-btn ghost"
                  disabled={formSaving}
                  onClick={() => setFormModal(undefined)}>
                  取消
                </button>
                <button type="submit" className="mcp-btn primary" disabled={formSaving}>
                  {formSaving ? '正在保存…' : formModal.serverId ? '保存修改' : '保存为未信任草稿'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {importOpen && (
        <div className="mcp-modal-overlay" role="dialog" aria-modal="true" aria-labelledby="mcp-import-title">
          <div className="mcp-modal-card">
            <div className="mcp-modal-header">
              <h3 id="mcp-import-title">导入 MCP JSON 配置</h3>
              <button
                type="button"
                className="icon-close-btn"
                aria-label="关闭导入弹窗"
                disabled={importSaving}
                onClick={() => {
                  setImportRaw('');
                  setImportOpen(false);
                }}>
                ✕
              </button>
            </div>

            <form className="mcp-form" onSubmit={submitImportJson}>
              <p className="mcp-security-note">
                支持标准 <code>{'{"mcpServers": {...}}'}</code> 格式（stdio 与 Streamable HTTP）。导入仅创建<strong>停用且未信任</strong>的草稿，不会自动执行任何程序，原始 JSON 也不会保存到 localStorage 或日志。
              </p>

              <label className="mcp-field">
                <span>MCP JSON 配置</span>
                <textarea
                  rows={8}
                  aria-label="MCP JSON 配置"
                  value={importRaw}
                  placeholder={'{\n  "mcpServers": {\n    "example": {\n      "command": "/opt/homebrew/bin/python3",\n      "args": ["/path/to/server.py"]\n    }\n  }\n}'}
                  onChange={(e) => {
                    setImportRaw(e.target.value);
                    setImportError('');
                  }}
                />
              </label>

              {importPreview && (
                importPreview.ok ? (
                  <div className="mcp-import-preview" aria-label="导入预览">
                    <h4>解析预览（共 {importPreview.servers.length} 个草稿）</h4>
                    <ul>
                      {importPreview.servers.map((item) => (
                        <li key={item.name}>
                          <div className="mcp-import-preview-head">
                            <strong>{item.name}</strong>
                            <span className="badge badge-transport">
                              {item.transport === 'stdio' ? 'stdio' : 'Streamable HTTP'}
                            </span>
                            <span className="mcp-pill tone-muted">导入后：已停用 · 待信任</span>
                          </div>
                          <div className="mcp-import-preview-meta">
                            <code>
                              {item.summaryTarget}
                              {item.args.length > 0 ? ` ${item.args.join(' ')}` : ''}
                            </code>
                            {item.cwd && <span> · 工作目录：{item.cwd}</span>}
                          </div>
                          <div className="mcp-import-preview-secrets">
                            {item.secretKeys.length > 0
                              ? `检测到 ${item.secretKeys.length} 个凭据键（将隔离存入钥匙串，不回显明文）：${item.secretKeys.join('、')}`
                              : '未检测到需要存入钥匙串的凭据键。'}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : (
                  <p role="alert" className="mcp-banner danger">{importPreview.error}</p>
                )
              )}

              {importError && <p role="alert" className="mcp-banner danger">{importError}</p>}

              <div className="mcp-modal-actions">
                <button
                  type="button"
                  className="mcp-btn ghost"
                  disabled={importSaving}
                  onClick={() => {
                    setImportRaw('');
                    setImportOpen(false);
                  }}>
                  取消
                </button>
                <button
                  type="submit"
                  className="mcp-btn primary"
                  disabled={importSaving || !importPreview?.ok}>
                  {importSaving ? '正在导入…' : '确认导入为未信任草稿'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </section>
  );
}
