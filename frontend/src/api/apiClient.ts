import { invoke } from '@tauri-apps/api/core';
import {
  parseRunEvent,
  type ArtifactSummary,
  type ParseWarning,
  type RunEvent,
  type StoredCitation,
  type ToolCategory,
} from './chatEvents';
import { readSse } from './sse';

export type { ArtifactSummary, ParseWarning, StoredCitation, ToolCategory };
export type RetrievalMode = 'keyword' | 'vector' | 'hybrid';
export type CoverageMode = 'global' | 'per_document';
export type ChannelStatus = 'hybrid' | 'keyword_only' | 'vector_only' | 'partial' | 'unavailable';
export type EvidenceStatus = 'evidence' | 'no_evidence' | 'not_returned' | 'not_assessed' | 'unavailable';

export type ChannelRank = {
  channel: 'keyword' | 'vector';
  rank: number;
  score: number;
};

export type KnowledgeSearchHit = {
  evidence_id: string;
  chunk_id: string;
  chunk_set_id: number;
  chunk_index: number;
  document_id: string;
  document_version_id: string;
  textbook_name: string;
  format: 'pdf' | 'docx' | 'md';
  section_path: string[];
  location_summary: string;
  location_precision: string;
  location_status: string;
  physical_pages: number[];
  page_labels: string[];
  line_range: [number, number] | null;
  paragraph_numbers: number[];
  warnings: ParseWarning[];
  historical: boolean;
  parse_spec_current: boolean;
  text_fidelity: string;
  body_text: string;
  fusion_score: number;
  channel_ranks: ChannelRank[];
  keyword_matched_in: ('body' | 'title')[];
};

export type DocumentCoverageView = {
  document_id: string;
  textbook_name: string | null;
  document_version_id: string | null;
  chunk_set_id: number | null;
  keyword_indexed: boolean;
  vector_indexed: boolean | null;
  searched_channels: ('keyword' | 'vector')[];
  evidence_status: EvidenceStatus;
  returned_count: number;
};

export type KnowledgeSearchResponse = {
  query: string;
  retrieval_mode: RetrievalMode;
  coverage_mode: CoverageMode;
  channel_status: ChannelStatus;
  hits: KnowledgeSearchHit[];
  documents: DocumentCoverageView[];
  diagnostic: { code: string; message: string } | null;
  timings: {
    scope_ms: number;
    fts_ms: number;
    vector_ms: number;
    query_embedding_ms: number;
    fusion_ms: number;
    source_ms: number;
  };
};

export type KnowledgeQueryOptions = {
  enabled: boolean;
  document_ids?: string[] | null;
  retrieval_mode?: RetrievalMode;
  coverage_mode?: CoverageMode;
  top_k?: number;
};

export type McpRunOptions = {
  enabled: boolean;
  allow_external_data_egress: boolean;
};

export type SkillsRunOptions = {
  enabled: boolean;
  selected_skill_ids?: string[];
};

export type SkillSourceType = 'builtin' | 'local' | 'github';

export type SkillDependencyStatus = {
  ready: boolean;
  missing_python_packages: string[];
  missing_node_tools: string[];
  missing_system_commands: string[];
  can_auto_prepare: boolean;
  summary: string;
};

export type SkillView = {
  skill_id: string;
  name: string;
  description: string;
  source_type: SkillSourceType;
  source_url: string | null;
  revision: string;
  enabled: boolean;
  ready: boolean;
  dependency_status: SkillDependencyStatus;
  warnings: string[];
  created_at: string;
  updated_at: string;
};

export type SkillResourceEntry = {
  relative_path: string;
  kind: 'script' | 'reference' | 'asset' | 'other';
  size_bytes: number;
};

export type SkillDetailView = SkillView & {
  license: string | null;
  compatibility: string | null;
  allowed_tools: string[];
  extra_frontmatter: Record<string, unknown>;
  metadata: Record<string, unknown>;
  instructions: string;
  resources: SkillResourceEntry[];
};

export type SkillImportCandidate = {
  relative_path: string;
  name: string;
  description: string;
  license: string | null;
  compatibility: string | null;
  resource_count: number;
  dependency_status: SkillDependencyStatus;
  required_python_packages: string[];
  required_node_tools: string[];
  required_system_commands: string[];
  warnings: string[];
  compatibility_notes: string[];
  existing_skill_id: string | null;
};

export type SkillImportPreview = {
  preview_id: string;
  source_kind: 'github' | 'local_dir' | 'local_zip';
  source_url: string;
  resolved_ref: string | null;
  commit_sha: string | null;
  subpath: string | null;
  candidates: SkillImportCandidate[];
};

export type ToolProbe = {
  available: boolean;
  path: string | null;
  version: string | null;
  source: string | null;
};

export type RuntimeStatus = {
  runtime_dir: string;
  venv_path: string;
  venv_ready: boolean;
  venv_python: string | null;
  installed_python_packages: Record<string, string>;
  puppeteer_config_path: string | null;
  uv: ToolProbe;
  python: ToolProbe;
  node: ToolProbe;
  npm: ToolProbe;
  npx: ToolProbe;
  git: ToolProbe;
  chrome: ToolProbe;
  mermaid_cli: ToolProbe;
  shell: ToolProbe;
  warnings: string[];
};

export type WorkspaceFileView = {
  file_id: string;
  session_id: string;
  original_name: string;
  relative_path: string;
  media_type: string;
  size_bytes: number;
  created_at: string;
  file_exists?: boolean;
};

export type StoredArtifact = {
  artifact_id: string;
  session_id: string;
  run_id: string;
  filename: string;
  title: string | null;
  relative_path: string;
  media_type: string;
  size_bytes: number;
  status: 'staged' | 'published' | 'discarded';
  created_at: string;
  file_exists?: boolean;
};

export type StoredSkillActivation = {
  activation_id: string;
  skill_id: string;
  name_snapshot: string;
  revision: string;
  activated_at: string;
};

export type SessionWorkspaceView = {
  session_id: string;
  files: WorkspaceFileView[];
  artifacts: StoredArtifact[];
};

export type McpTransportKind = 'stdio' | 'streamable_http';
export type McpRuntimeState = 'disconnected' | 'connecting' | 'connected' | 'disconnecting' | 'error';
export type McpTrustState = 'untrusted' | 'trusted';
export type McpToolPermissionState = 'denied' | 'allowed';

export type McpConfigEntryView =
  | { kind: 'plain'; value: string }
  | { kind: 'secret'; secret_key: string; configured: boolean };

export type McpConfigEntryRequest =
  | { kind: 'plain'; value: string }
  | { kind: 'secret'; secret_key: string };

export type McpServerRuntime = {
  state: McpRuntimeState;
  generation: number;
  protocol_version: string | null;
  server_name_reported: string | null;
  server_version_reported: string | null;
  tool_count: number;
  catalog_version: number;
  catalog_stale: boolean;
  tools_truncated: boolean;
  last_error_code: string | null;
  last_error_message: string | null;
  stderr_tail: string[];
  connected_at: string | null;
};

export type McpServerView = {
  server_id: string;
  name: string;
  transport: McpTransportKind;
  command: string | null;
  resolved_command: string | null;
  executable_exists: boolean | null;
  args: string[];
  cwd: string | null;
  url: string | null;
  allow_private_network: boolean;
  private_network_target: boolean;
  high_risk_command: boolean;
  high_risk_reasons: string[];
  env_config: Record<string, McpConfigEntryView>;
  header_config: Record<string, McpConfigEntryView>;
  credential_ref: string | null;
  credentials_complete: boolean;
  missing_secret_keys: string[];
  enabled: boolean;
  trust_state: McpTrustState;
  trust_valid: boolean;
  config_revision: number;
  trusted_revision: number | null;
  trusted_fingerprint: string | null;
  current_fingerprint: string;
  discovered_tool_count: number;
  allowed_tool_count: number;
  effective_allowed_tool_count: number;
  runtime: McpServerRuntime;
  created_at: string;
  updated_at: string;
};

export type McpToolView = {
  tool_id: string;
  server_id: string;
  original_tool_name: string;
  title: string | null;
  description: string;
  input_schema: Record<string, unknown>;
  annotations: Record<string, unknown>;
  definition_hash: string;
  schema_compatible: boolean;
  incompatibility_reason: string | null;
  permission_state: McpToolPermissionState;
  approved_definition_hash: string | null;
  approved_config_revision: number | null;
  definition_changed: boolean;
  is_destructive: boolean;
  present_in_last_discovery: boolean;
  effective_allowed: boolean;
  discovered_at: string;
  updated_at: string;
};

export type McpSummary = {
  total_servers: number;
  enabled_servers: number;
  connected_servers: number;
  error_servers: number;
  effective_allowed_tools: number;
  max_tools_per_run: number;
  exceeds_tool_limit: boolean;
};

export type SaveMcpServerRequest = {
  server_id: string | null;
  name: string;
  transport: McpTransportKind;
  command: string | null;
  args: string[];
  cwd: string | null;
  clear_cwd: boolean;
  url: string | null;
  allow_private_network: boolean;
  env_config: Record<string, McpConfigEntryRequest>;
  header_config: Record<string, McpConfigEntryRequest>;
  secret_values: Record<string, string>;
  remove_secret_keys: string[];
};

export type ImportMcpServersResult = {
  servers: McpServerView[];
  warnings: string[];
};

export type SetMcpToolPermissionRequest = {
  server_id: string;
  tool_id: string;
  permission_state: McpToolPermissionState;
  expected_definition_hash: string | null;
  expected_config_revision: number | null;
};

export type ToolCallSummary = {
  tool_call_id: string;
  tool_name: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  summary: string | null;
  error_code: string | null;
  source?: 'native' | 'mcp';
  tool_category?: ToolCategory | null;
  recoverable?: boolean;
  server_id?: string | null;
  server_name?: string | null;
  original_tool_name?: string | null;
  duration_ms?: number | null;
};

type BackendConnection = {
  base_url: string;
  ui_token: string;
  siliconflow_restore_error?: string;
};

export type BackendStatus = {
  status: 'ready';
  database: string;
  model_configured: boolean;
  siliconflow_configured?: boolean;
};

export type Session = { session_id: string; title: string; created_at: string };
export type GrantScope = 'file' | 'directory';
export type FileGrant = {
  grant_id: string;
  session_id: string;
  scope: GrantScope;
  display_name: string;
  operations: ['read'];
};
export type DocumentSummary = {
  document_id: string;
  name: string;
  format: 'pdf' | 'docx' | 'md';
  management_status: 'imported';
  created_at: string;
  updated_at: string;
  version_count: number;
  latest_size_bytes: number | null;
};
export type DocumentVersion = {
  document_version_id: string;
  content_sha256: string;
  size_bytes: number;
  created_at: string;
};
export type DocumentDetails = DocumentSummary & { versions: DocumentVersion[] };
export type IndexJobStatus = 'queued' | 'running' | 'retry_wait' | 'succeeded' | 'failed' | 'cancelled';
export type IndexJobStage = 'parsing' | 'chunking' | 'keyword_indexing' | 'embedding' | 'publishing';
export type PublishedIndexReadiness = 'ready_keyword' | 'ready_hybrid' | 'unsearchable';
export type IndexJobState = {
  job_id: string;
  document_id: string;
  document_version_id: string;
  status: IndexJobStatus;
  stage: IndexJobStage;
  completed_work: number;
  total_work: number | null;
  error_code: string | null;
  retry_count: number;
  next_retry_at: number | null;
  cancel_requested_at: number | null;
};
export type DocumentIndex = {
  document_id: string;
  document_version_id: string;
  published_readiness: PublishedIndexReadiness | null;
  job: IndexJobState | null;
};
export type SessionSummary = Omit<Session, 'created_at'> & {
  updated_at: string;
  active_run_id: string | null;
  active_run_status: ActiveRunState | null;
};
export type Message = {
  message_id: string;
  session_id: string;
  run_id: string;
  role: 'user' | 'assistant';
  content: string;
  created_at: string;
  status: 'committed';
};
export type ApiClient = Awaited<ReturnType<typeof connectApiClient>>;

type TerminalRunState = 'completed' | 'failed' | 'cancelled';
type ActiveRunState = 'running' | 'awaiting_approval' | 'cancelling';
export type RunState = ActiveRunState | TerminalRunState;
export type RunStatus = {
  run_id: string;
  session_id: string;
  status: RunState;
  started_at: string;
  finished_at: string | null;
  error_code: string | null;
  pending_approval: null;
};
export type RunSummary = Omit<RunStatus, 'pending_approval'> & {
  user_message_id: string;
  error_message?: string | null;
  tool_calls?: ToolCallSummary[];
  citations?: StoredCitation[];
  artifacts?: StoredArtifact[];
  skill_activations?: StoredSkillActivation[];
};
type RunPage = { items: RunSummary[]; next_cursor: string | null };
type CancelResult =
  | { run_id: string; status: 'cancelling' }
  | { run_id: string; status: TerminalRunState; already_finished: true };

export function isTerminal(status: string): status is TerminalRunState {
  return status === 'completed' || status === 'failed' || status === 'cancelled';
}

export class ApiRequestError extends Error {
  constructor(message: string, readonly code?: string, readonly runId?: string) {
    super(message);
  }
}

export function extractErrorCode(error: unknown): string | undefined {
  if (error instanceof ApiRequestError) return error.code;
  if (typeof error === 'string') return error;
  if (error && typeof error === 'object' && 'code' in error && typeof error.code === 'string') {
    return error.code;
  }
  return undefined;
}

export function mcpErrorMessage(error: unknown, fallback: string): string {
  switch (extractErrorCode(error)) {
    case 'MCP_TRUST_CANCELLED': return '已取消信任确认，服务仍保持未信任状态。';
    case 'MCP_TRUST_DIALOG_FAILED': return '无法弹出原生信任确认框，请重试或检查桌面窗口状态。';
    case 'MCP_CREDENTIAL_REJECTED': return '凭据格式无效（不能为空、不能包含换行或空字符，且需与已声明的凭据键对应）。';
    case 'MCP_CREDENTIAL_REPLAY_FAILED': return '凭据已写入钥匙串，但未能同步到后端内存，请稍后重试连接。';
    case 'MCP_KEYCHAIN_CLEANUP_FAILED': return '服务配置已删除，但清理 macOS 钥匙串中的凭据失败，请在钥匙串访问中手动确认。';
    case 'KEYCHAIN_ACCESS_FAILED': return '无法访问 macOS 钥匙串，请检查系统权限后重试。';
    case 'KEYCHAIN_CREDENTIAL_CORRUPTED': return 'macOS 钥匙串中的 MCP 凭据已损坏，请重新保存凭据。';
    case 'MCP_INVALID_CONFIG': return 'MCP 服务配置无效，请检查字段格式、URL 或命令行参数（禁止在明文参数/环境变量中夹带密钥）。';
    case 'MCP_NETWORK_POLICY_VIOLATION': return 'HTTP 目标地址违反安全网络策略（非本地回环必须使用 HTTPS，私网地址需显式允许）。';
    case 'MCP_EXECUTABLE_NOT_FOUND': return '未找到指定路径的可执行程序，请填写准确的绝对路径或检查本机环境。';
    case 'MCP_SERVER_NAME_CONFLICT': return '已存在同名的 MCP 服务，请更换服务名称。';
    case 'MCP_SERVER_NOT_FOUND': return '该 MCP 服务已不存在，请刷新列表。';
    case 'MCP_TOOL_NOT_FOUND': return '该 MCP 工具已不存在，请刷新工具列表。';
    case 'MCP_UNTRUSTED_SERVER': return '该 MCP 服务尚未完成信任确认或关键配置已变更，请先通过原生弹窗确认信任。';
    case 'MCP_TRUST_STALE': return '确认期间服务配置发生了变更，请核对最新配置后重新确认信任。';
    case 'MCP_CREDENTIALS_MISSING': return '该 MCP 服务缺少必需的钥匙串凭据，请编辑服务并补充凭据后再连接。';
    case 'MCP_ACTIVE_SERVER_LIMIT': return '同时连接的 MCP 服务数量已达上限（8 个），请先断开其他服务。';
    case 'MCP_CONNECT_TIMEOUT': return '连接 MCP 服务超时，请检查本地程序是否能正常响应或网络是否可达。';
    case 'MCP_CONNECT_FAILED': return '无法连接到 MCP 服务，请查看脱敏诊断信息排查原因。';
    case 'MCP_DISCOVERY_FAILED': return '获取 MCP 工具列表失败，请稍后重试。';
    case 'MCP_SCHEMA_INCOMPATIBLE': return '该工具的参数 Schema 不兼容，无法授权启用。';
    case 'MCP_DESTRUCTIVE_TOOL_BLOCKED': return '该工具被标记为破坏性操作，当前版本不允许授权启用。';
    case 'MCP_PERMISSION_DENIED': return 'MCP 工具未获得有效授权，或服务配置/工具定义已发生变化。';
    case 'MCP_SERVER_UNAVAILABLE': return 'MCP 服务当前未启用、未连接或正在刷新工具目录。';
    case 'MCP_SERVER_BUSY': return '该 MCP 服务当前请求排队已满，请稍后重试。';
    case 'MCP_TOOL_TIMEOUT': return 'MCP 工具调用超时，外部执行结果未知。';
    case 'MCP_TOOL_RETURNED_ERROR': return 'MCP 工具返回了错误结果。';
    case 'MCP_PROTOCOL_ERROR': return 'MCP 协议通信失败或连接中断。';
    case 'MCP_RESPONSE_TOO_LARGE': return 'MCP 响应体积超过安全上限，已中止读取。';
    case 'MCP_MALFORMED_RESPONSE': return 'MCP 服务返回了畸形或无法解析的数据。';
    case 'MCP_CALL_ABORTED': return '同轮前置工具调用已失败，已中止后续外部 MCP 调用。';
    case 'MCP_TOOL_LIMIT_EXCEEDED': return '当前有效授权工具数量或 Schema 体积超过单轮上限（32 个 / 128 KiB），请前往 MCP 服务页面精简授权工具。';
    case 'MCP_EGRESS_AUTHORIZATION_REQUIRED': return '当前对话包含受保护的本地文件、教材检索或不可验证的历史上下文，需显式授权本轮数据外发后才能启用 MCP。';
    case 'RUN_TIMEOUT': return '本轮对话执行总时长已超过上限。';
    case 'SKILL_REPEAT_FAILURE_LIMIT': return '脚本或工具连续出现相同错误达到上限，请调整需求或检查依赖后重试。';
    case 'SKILL_INVALID_SOURCE': return '来源地址或压缩包格式无效，请检查 GitHub 链接或本地文件。';
    case 'SKILL_NOT_FOUND': return '未在来源中发现有效的 SKILL.md 技能目录。';
    case 'SKILL_IMPORT_FAILED': return '下载或解包技能来源失败，请检查网络连接或仓库地址。';
    case 'SKILL_PREVIEW_EXPIRED': return '导入预览已过期，请重新获取技能列表。';
    case 'BACKEND_SIDECAR_UNAVAILABLE':
    case 'BACKEND_SHUTTING_DOWN':
    case 'SIDECAR_NOT_READY': return '桌面后端暂不可用，请检查连接状态后重试。';
    default:
      if (error instanceof ApiRequestError && error.message) return error.message;
      return fallback;
  }
}

// ---------------------------------------------------------------------------
// Privileged MCP Tauri IPC Wrappers (never hold or expose Host Token in WebView)
// ---------------------------------------------------------------------------

export function mcpSaveServer(request: SaveMcpServerRequest): Promise<McpServerView> {
  return invoke<McpServerView>('mcp_save_server', { request });
}

export function mcpImportServersJson(rawJson: string): Promise<ImportMcpServersResult> {
  return invoke<ImportMcpServersResult>('mcp_import_servers_json', {
    request: { raw_json: rawJson },
  });
}

export function mcpTrustServer(serverId: string, enable = false): Promise<McpServerView> {
  return invoke<McpServerView>('mcp_trust_server', {
    request: { server_id: serverId, enable },
  });
}

export function mcpRevokeTrust(serverId: string): Promise<McpServerView> {
  return invoke<McpServerView>('mcp_revoke_trust', {
    request: { server_id: serverId },
  });
}

export function mcpDeleteServer(serverId: string): Promise<void> {
  return invoke<void>('mcp_delete_server', {
    request: { server_id: serverId },
  });
}

export function mcpConnectServer(serverId: string): Promise<McpServerView> {
  return invoke<McpServerView>('mcp_connect_server', {
    request: { server_id: serverId },
  });
}

export function mcpDisconnectServer(serverId: string): Promise<McpServerView> {
  return invoke<McpServerView>('mcp_disconnect_server', {
    request: { server_id: serverId },
  });
}

export function mcpRefreshTools(serverId: string): Promise<McpToolView[]> {
  return invoke<McpToolView[]>('mcp_refresh_tools', {
    request: { server_id: serverId },
  });
}

export function mcpSetServerEnabled(serverId: string, enabled: boolean): Promise<McpServerView> {
  return invoke<McpServerView>('mcp_set_server_enabled', {
    request: { server_id: serverId, enabled },
  });
}

export function mcpSetToolPermission(request: SetMcpToolPermissionRequest): Promise<McpToolView> {
  return invoke<McpToolView>('mcp_set_tool_permission', { request });
}

// ---------------------------------------------------------------------------
// Privileged Workspace, Artifact, Runtime & Skills Tauri IPC Wrappers
// ---------------------------------------------------------------------------

export function selectAndImportWorkspaceFiles(sessionId: string): Promise<WorkspaceFileView[]> {
  return invoke<WorkspaceFileView[]>('select_and_import_workspace_files', {
    request: { session_id: sessionId },
  });
}

export function importDroppedWorkspaceFiles(sessionId: string, paths: string[]): Promise<WorkspaceFileView[]> {
  return invoke<WorkspaceFileView[]>('import_dropped_workspace_files', {
    request: { session_id: sessionId, paths },
  });
}

export function deleteWorkspaceFile(sessionId: string, fileId: string): Promise<void> {
  return invoke<void>('delete_workspace_file', {
    request: { session_id: sessionId, file_id: fileId },
  });
}

export function openArtifact(sessionId: string, artifactId: string): Promise<void> {
  return invoke<void>('open_artifact', {
    request: { session_id: sessionId, artifact_id: artifactId },
  });
}

export function revealArtifactInFinder(sessionId: string, artifactId: string): Promise<void> {
  return invoke<void>('reveal_artifact_in_finder', {
    request: { session_id: sessionId, artifact_id: artifactId },
  });
}

export function exportArtifact(sessionId: string, artifactId: string): Promise<{ exported: boolean; filename: string } | null> {
  return invoke<{ exported: boolean; filename: string } | null>('export_artifact', {
    request: { session_id: sessionId, artifact_id: artifactId },
  });
}

export function runtimePrepare(request: {
  python_packages?: string[];
  install_mermaid_cli?: boolean;
  ensure_python_venv?: boolean;
  timeout_seconds?: number;
}): Promise<{ ok: boolean; status: RuntimeStatus; steps: { step: string; ok: boolean; message: string; duration_ms: number }[] }> {
  return invoke('runtime_prepare', { request });
}

export function skillsPreviewGithub(url: string, refName?: string, subpath?: string): Promise<SkillImportPreview> {
  return invoke<SkillImportPreview>('skills_preview_github', {
    request: {
      url,
      ...(refName?.trim() ? { ref_name: refName.trim() } : {}),
      ...(subpath?.trim() ? { subpath: subpath.trim() } : {}),
    },
  });
}

export function skillsPickAndPreviewLocal(mode: 'directory' | 'zip'): Promise<SkillImportPreview | null> {
  return invoke<SkillImportPreview | null>('skills_pick_and_preview_local', {
    request: { mode },
  });
}

export function skillsInstallFromPreview(
  previewId: string,
  selectedRelativePaths: string[],
  enabled = true,
): Promise<{ skills: SkillView[] }> {
  return invoke<{ skills: SkillView[] }>('skills_install_from_preview', {
    request: {
      preview_id: previewId,
      selected_relative_paths: selectedRelativePaths,
      enabled,
      replace_existing: true,
    },
  });
}

export function skillsSetEnabled(skillId: string, enabled: boolean): Promise<SkillView> {
  return invoke<SkillView>('skills_set_enabled', {
    request: { skill_id: skillId, enabled },
  });
}

export function skillsUpdate(skillId: string): Promise<SkillView> {
  return invoke<SkillView>('skills_update', {
    request: { skill_id: skillId },
  });
}

export function skillsPrepare(skillId: string): Promise<{
  ok: boolean;
  dependency_status: SkillDependencyStatus;
  skill: SkillView;
}> {
  return invoke('skills_prepare', {
    request: { skill_id: skillId },
  });
}

export function skillsDelete(skillId: string): Promise<void> {
  return invoke<void>('skills_delete', {
    request: { skill_id: skillId },
  });
}

async function requireSuccess(response: Response) {
  if (response.ok) return;
  const body = await response.json().catch(() => null);
  const error = body?.error;
  throw new ApiRequestError(
    typeof error?.message === 'string' ? error.message : `请求失败（HTTP ${response.status}）。`,
    typeof error?.code === 'string' ? error.code : undefined,
    typeof error?.run_id === 'string' ? error.run_id : undefined,
  );
}

function indexJobsPath(documentId: string, versionId: string) {
  return `/knowledge/documents/${encodeURIComponent(documentId)}/versions/${encodeURIComponent(versionId)}/index-jobs`;
}

function requireIndexJob(job: IndexJobState, documentId: string, versionId: string, jobId?: string) {
  if (job?.document_id !== documentId || job.document_version_id !== versionId || typeof job.job_id !== 'string'
    || (jobId !== undefined && job.job_id !== jobId)) {
    throw new Error('后端返回的索引任务标识无效。');
  }
  return job;
}

function isBackendStatus(value: unknown): value is BackendStatus {
  return typeof value === 'object'
    && value !== null
    && 'status' in value && value.status === 'ready'
    && 'database' in value && typeof value.database === 'string'
    && 'model_configured' in value && typeof value.model_configured === 'boolean'
    && (!('siliconflow_configured' in value) || typeof value.siliconflow_configured === 'boolean');
}

export async function connectApiClient() {
  let connection: BackendConnection;
  try {
    connection = await invoke<BackendConnection>('get_backend_connection');
  } catch (error) {
    if (error === 'MODEL_CREDENTIAL_REPLAY_FAILED') {
      throw new Error('DeepSeek 凭据恢复失败，模型未配置。可重试连接，或退出应用后重新配置。');
    }
    if (error === 'KEYCHAIN_ACCESS_FAILED') {
      throw new Error('无法访问 macOS Keychain 读取 DeepSeek 凭据，请检查钥匙串权限后重试。');
    }
    if (error === 'KEYCHAIN_CREDENTIAL_CORRUPTED') {
      throw new Error('macOS Keychain 中的 DeepSeek 凭据已损坏，请清理或重新配置。');
    }
    if (error === 'BACKEND_DATABASE_LOCKED') {
      throw new Error('检测到另一个 KnowledgeAgent 实例正在运行并占用数据目录，请先关闭已运行的应用后重试。');
    }
    if (error === 'BACKEND_SIDECAR_UNAVAILABLE' || error === 'BACKEND_SHUTTING_DOWN') {
      throw new Error('桌面后端暂不可用，请点击「重试」重新连接。');
    }
    if (error === 'BACKEND_NOT_CONFIGURED') {
      throw new Error('此构建尚未配置后端连接。');
    }
    throw new Error('无法获取桌面宿主的连接配置。请检查本地开发配置。');
  }

  function request(
    path: string, method: 'GET' | 'POST', signal: AbortSignal,
    { accept = 'application/json', body }: { accept?: string; body?: object } = {},
  ) {
    return fetch(`${connection.base_url}/api/v1${path}`, {
      method,
      headers: {
        Accept: accept,
        Authorization: `Bearer ${connection.ui_token}`,
        ...(body && { 'Content-Type': 'application/json' }),
      },
      body: body && JSON.stringify(body),
      cache: 'no-store',
      credentials: 'omit',
      redirect: 'error',
      signal,
    });
  }

  async function readRun(path: string, sessionId: string, signal: AbortSignal, runId?: string): Promise<RunStatus> {
    const run = await readJson<RunStatus>(path, signal);
    if (run?.session_id !== sessionId || typeof run.run_id !== 'string' || (runId && run.run_id !== runId)
      || !(isTerminal(run.status) || ['running', 'cancelling', 'awaiting_approval'].includes(run.status))) {
      throw new Error('后端返回的运行标识或状态无效。');
    }
    return run;
  }

  async function readJson<T>(path: string, signal: AbortSignal): Promise<T> {
    const response = await request(path, 'GET', AbortSignal.any([signal, AbortSignal.timeout(5000)]));
    await requireSuccess(response);
    return await response.json();
  }

  async function indexAction(
    path: string, documentId: string, versionId: string, signal: AbortSignal, body?: object, jobId?: string,
  ): Promise<IndexJobState> {
    const response = await request(path, 'POST', AbortSignal.any([signal, AbortSignal.timeout(5000)]), { body });
    await requireSuccess(response);
    return requireIndexJob(await response.json(), documentId, versionId, jobId);
  }

  return {
    siliconflowRestoreError: connection.siliconflow_restore_error,
    async getStatus(signal: AbortSignal): Promise<BackendStatus> {
      let response: Response;
      try {
        response = await request('/status', 'GET', AbortSignal.any([signal, AbortSignal.timeout(5000)]));
      } catch {
        throw new Error('无法连接后端。请确认 FastAPI 服务已启动。');
      }

      if (response.status === 401 || response.status === 403) {
        throw new Error(`后端拒绝了连接凭证（HTTP ${response.status}）。`);
      }
      if (!response.ok) {
        throw new Error(`后端状态请求失败（HTTP ${response.status}）。`);
      }

      const status: unknown = await response.json().catch(() => null);
      if (!isBackendStatus(status)) {
        throw new Error('后端返回了无效的状态响应。');
      }
      return status;
    },
    async createSession(title: string, signal: AbortSignal): Promise<Session> {
      const response = await request('/sessions', 'POST', AbortSignal.any([signal, AbortSignal.timeout(5000)]), { body: { title } });
      await requireSuccess(response);
      return await response.json();
    },
    async renameSession(sessionId: string, title: string, signal: AbortSignal): Promise<Session> {
      const response = await request(`/sessions/${encodeURIComponent(sessionId)}/rename`, 'POST',
        AbortSignal.any([signal, AbortSignal.timeout(5000)]), { body: { title } });
      await requireSuccess(response);
      return await response.json();
    },
    async deleteSession(sessionId: string, signal: AbortSignal): Promise<void> {
      const response = await request(`/sessions/${encodeURIComponent(sessionId)}/delete`, 'POST',
        AbortSignal.any([signal, AbortSignal.timeout(5000)]));
      await requireSuccess(response);
      if (response.status !== 204) throw new Error('后端返回了无效的会话删除结果。');
    },
    async searchKnowledge(
      params: {
        query: string;
        top_k?: number;
        document_ids?: string[] | null;
        retrieval_mode?: RetrievalMode;
        coverage_mode?: CoverageMode;
      },
      signal: AbortSignal,
    ): Promise<KnowledgeSearchResponse> {
      const response = await request('/knowledge/search', 'POST',
        AbortSignal.any([signal, AbortSignal.timeout(30000)]), { body: params });
      await requireSuccess(response);
      return await response.json();
    },
    listSessions(signal: AbortSignal): Promise<SessionSummary[]> {
      return readJson('/sessions', signal);
    },
    listMessages(sessionId: string, signal: AbortSignal): Promise<Message[]> {
      return readJson(`/sessions/${encodeURIComponent(sessionId)}/messages`, signal);
    },
    async listGrants(sessionId: string, signal: AbortSignal): Promise<FileGrant[]> {
      const grants = await readJson<FileGrant[]>(`/sessions/${encodeURIComponent(sessionId)}/grants`, signal);
      if (!Array.isArray(grants) || grants.some((grant) => grant?.session_id !== sessionId)) {
        throw new Error('后端返回的授权会话无效。');
      }
      return grants;
    },
    listDocuments(signal: AbortSignal): Promise<DocumentSummary[]> {
      return readJson('/knowledge/documents', signal);
    },
    listMcpServers(signal: AbortSignal): Promise<McpServerView[]> {
      return readJson('/mcp/servers', signal);
    },
    listMcpServerTools(serverId: string, signal: AbortSignal): Promise<McpToolView[]> {
      return readJson(`/mcp/servers/${encodeURIComponent(serverId)}/tools`, signal);
    },
    getMcpSummary(signal: AbortSignal): Promise<McpSummary> {
      return readJson('/mcp/summary', signal);
    },
    listSkills(signal: AbortSignal): Promise<SkillView[]> {
      return readJson('/skills', signal);
    },
    getSkillDetail(skillId: string, signal: AbortSignal): Promise<SkillDetailView> {
      return readJson(`/skills/${encodeURIComponent(skillId)}`, signal);
    },
    getRuntimeStatus(signal: AbortSignal): Promise<RuntimeStatus> {
      return readJson('/runtime/status', signal);
    },
    getSessionWorkspace(sessionId: string, signal: AbortSignal): Promise<SessionWorkspaceView> {
      return readJson(`/sessions/${encodeURIComponent(sessionId)}/workspace`, signal);
    },
    async fetchArtifactBlob(sessionId: string, artifactId: string, signal: AbortSignal): Promise<Blob> {
      const response = await request(
        `/sessions/${encodeURIComponent(sessionId)}/artifacts/${encodeURIComponent(artifactId)}/content`,
        'GET',
        AbortSignal.any([signal, AbortSignal.timeout(15000)]),
        { accept: '*/*' },
      );
      await requireSuccess(response);
      return await response.blob();
    },
    getDocument(documentId: string, signal: AbortSignal): Promise<DocumentDetails> {
      return readJson(`/knowledge/documents/${encodeURIComponent(documentId)}`, signal);
    },
    async getDocumentIndex(documentId: string, signal: AbortSignal): Promise<DocumentIndex> {
      const index = await readJson<DocumentIndex>(`/knowledge/documents/${encodeURIComponent(documentId)}/index`, signal);
      if (index?.document_id !== documentId || typeof index.document_version_id !== 'string') {
        throw new Error('后端返回的教材索引标识无效。');
      }
      if (index.job !== null) requireIndexJob(index.job, documentId, index.document_version_id);
      return index;
    },
    submitDocumentIndex(documentId: string, versionId: string, signal: AbortSignal): Promise<IndexJobState> {
      return indexAction(indexJobsPath(documentId, versionId), documentId, versionId, signal,
        { allow_cloud_embedding: true });
    },
    cancelDocumentIndex(job: IndexJobState, signal: AbortSignal): Promise<IndexJobState> {
      const path = indexJobsPath(job.document_id, job.document_version_id);
      return indexAction(`${path}/${encodeURIComponent(job.job_id)}/cancel`, job.document_id, job.document_version_id,
        signal, undefined, job.job_id);
    },
    retryDocumentIndex(job: IndexJobState, signal: AbortSignal): Promise<IndexJobState> {
      const path = indexJobsPath(job.document_id, job.document_version_id);
      return indexAction(`${path}/${encodeURIComponent(job.job_id)}/retry`, job.document_id, job.document_version_id,
        signal, { allow_cloud_embedding: true });
    },
    async deleteDocument(documentId: string, signal: AbortSignal): Promise<void> {
      const path = `/knowledge/documents/${encodeURIComponent(documentId)}/delete`;
      const response = await request(path, 'POST', AbortSignal.any([signal, AbortSignal.timeout(5000)]));
      await requireSuccess(response);
      if (response.status !== 204) throw new Error('后端返回了无效的删除结果。');
    },
    async listRuns(sessionId: string, signal: AbortSignal): Promise<RunSummary[]> {
      const runs: RunSummary[] = [];
      const cursors = new Set<string>();
      const path = `/sessions/${encodeURIComponent(sessionId)}/runs`;
      let cursor: string | null = null;
      do {
        const query = new URLSearchParams({ limit: '100' });
        if (cursor) query.set('cursor', cursor);
        const page = await readJson<RunPage>(`${path}?${query}`, signal);
        runs.push(...page.items);
        cursor = page.next_cursor;
        if (cursor) {
          if (cursors.has(cursor)) throw new Error('后端返回了重复的分页游标。');
          cursors.add(cursor);
        }
      } while (cursor);
      return runs;
    },
    getRun(sessionId: string, runId: string, signal: AbortSignal): Promise<RunStatus> {
      return readRun(`/runs/${encodeURIComponent(runId)}`, sessionId, signal, runId);
    },
    lookupRun(sessionId: string, requestId: string, signal: AbortSignal): Promise<RunStatus> {
      const query = new URLSearchParams({ session_id: sessionId, client_request_id: requestId });
      return readRun(`/runs/lookup?${query}`, sessionId, signal);
    },
    async cancelRun(runId: string, signal: AbortSignal): Promise<CancelResult> {
      const response = await request(`/runs/${encodeURIComponent(runId)}/cancel`, 'POST',
        AbortSignal.any([signal, AbortSignal.timeout(5000)]));
      await requireSuccess(response);
      const result = await response.json();
      if (result?.run_id !== runId
        || !((response.status === 202 && result.status === 'cancelling')
          || (response.status === 200 && isTerminal(result.status) && result.already_finished === true))) {
        throw new Error('后端返回了无效的取消响应。');
      }
      return result;
    },
    async streamRun(
      sessionId: string, clientRequestId: string, message: string,
      signal: AbortSignal, onEvent: (event: RunEvent) => void,
      knowledge?: KnowledgeQueryOptions,
      mcp?: McpRunOptions,
      skills?: SkillsRunOptions,
      attachmentIds?: string[],
      artifactIds?: string[],
    ): Promise<void> {
      const response = await request(`/sessions/${encodeURIComponent(sessionId)}/runs/stream`, 'POST', signal, {
        accept: 'text/event-stream',
        body: {
          client_request_id: clientRequestId,
          message,
          ...(knowledge ? { knowledge } : {}),
          ...(mcp ? { mcp } : {}),
          ...(skills ? { skills } : {}),
          ...(attachmentIds && attachmentIds.length > 0 ? { attachment_ids: attachmentIds } : {}),
          ...(artifactIds && artifactIds.length > 0 ? { artifact_ids: artifactIds } : {}),
        },
      });
      await requireSuccess(response);
      if (!response.headers.get('Content-Type')?.startsWith('text/event-stream') || !response.body) {
        throw new Error('后端返回了无效的生成流。');
      }

      let runId: string | undefined;
      for await (const frame of readSse(response.body)) {
        const event = parseRunEvent(frame);
        if (!event || event.session_id !== sessionId) continue;
        if (event.type === 'run_started') {
          if (runId || event.data.client_request_id !== clientRequestId) continue;
          runId = event.run_id;
        }
        if (event.run_id !== runId) continue;
        onEvent(event);
        if (event.type === 'run_completed' || event.type === 'error' || event.type === 'run_cancelled') return;
      }
      throw new Error('生成流提前中断。');
    },
  };
}
