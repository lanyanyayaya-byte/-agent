import { isTauri } from '@tauri-apps/api/core';
import { getCurrentWebview } from '@tauri-apps/api/webview';
import { useCallback, useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import {
  ApiRequestError,
  deleteWorkspaceFile,
  exportArtifact,
  importDroppedWorkspaceFiles,
  isTerminal,
  mcpErrorMessage,
  openArtifact,
  revealArtifactInFinder,
  selectAndImportWorkspaceFiles,
  type ApiClient,
  type CoverageMode,
  type DocumentSummary,
  type FileGrant,
  type McpSummary,
  type Message,
  type RetrievalMode,
  type RunState,
  type RunSummary,
  type SessionSummary,
  type SessionWorkspaceView,
  type SkillView,
  type StoredArtifact,
  type StoredCitation,
  type StoredSkillActivation,
  type ToolCallSummary,
  type WorkspaceFileView,
} from './api/apiClient';
import type { ArtifactSummary, RunEvent, ToolCategory, ToolSource } from './api/chatEvents';
import GrantPanel from './GrantPanel';
import MarkdownAnswer from './MarkdownAnswer';

const MESSAGE_LIMIT = 20_000;
const POLL_INTERVAL_MS = 1000;
const SELECTED_SESSION_KEY = 'knowledgeagent.selected_session';
const MCP_ENABLED_STORAGE_KEY = 'knowledgeagent.mcp_enabled';
const SKILLS_ENABLED_STORAGE_KEY = 'knowledgeagent.skills_enabled';
const SKILLS_MODE_STORAGE_KEY = 'knowledgeagent.skills_mode';
const DEFAULT_SESSION_TITLE = /^(?:新对话|会话\s*\d+)$/;
const TITLE_FILLER_PREFIX =
  /^(?:你好(?:呀|啊)?[，,！!。.\s]*|请问一下|请问|请教一下|请教|请帮我|帮我|给我|能否|可以|麻烦|详细讲解一下|深入讲解一下|讲解一下|详细讲解|讲解|讲讲|讲一下|谈谈|说说|详细解释一下|解释一下|请解释|总结一下|请总结|介绍一下|请介绍|分析一下|推导一下|证明一下|梳理一下|对比不同教材中(?:关于)?|不同教材中(?:关于)?|告诉我|什么是|如何理解|怎么理解|关于)+\s*/;

export function deriveSessionTitle(question: string): string {
  const firstLine = question
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find(Boolean) ?? '';
  const base = firstLine.replace(/\s+/g, ' ').replace(/[？?。！!，,；;：:\s]+$/g, '').trim();
  if (!base) return '新对话';
  let cleaned = base;
  if (!/(?:自己|你是谁|你是什么)/.test(base)) {
    const stripped = base.replace(TITLE_FILLER_PREFIX, '').replace(/[？?。！!，,；;：:\s]+$/g, '').trim();
    if (Array.from(stripped).length >= 3) cleaned = stripped;
  }
  const chars = Array.from(cleaned);
  return chars.length > 18 ? `${chars.slice(0, 18).join('')}…` : cleaned;
}

type Turn = { user: Message; run?: RunSummary; assistant?: Message };
type ToolProgress = {
  id: string;
  name: string;
  state: 'running' | 'completed' | 'failed' | 'cancelled';
  summary?: string;
  errorCode?: string;
  source?: ToolSource;
  toolCategory?: ToolCategory | null;
  recoverable?: boolean;
  serverId?: string;
  serverName?: string;
  originalToolName?: string;
  durationMs?: number;
};
type ToolEvent = Extract<RunEvent, { type: 'tool_started' | 'tool_completed' | 'tool_failed' }>;
type ActivatedSkillBadge = { skill_id: string; name: string; revision?: string };

type EgressPrompt = {
  sessionId: string;
  question: string;
  currentRagRisk: boolean;
  currentFileGrantRisk: boolean;
  historicalRagRisk: boolean;
  historicalFileRisk: boolean;
  backendMessage?: string;
  backendForcedHistoryRisk?: boolean;
};

type LiveTurn = {
  requestId: string;
  question: string;
  runId?: string;
  draft: string;
  detached?: boolean;
  lookupComplete?: boolean;
  settling?: boolean;
  problem?: string;
};

type Conversation = Pick<SessionSummary, 'session_id' | 'title'> & {
  input: string;
  turns?: Turn[];
  activeRunId?: string;
  activeStatus?: RunState;
  toolsByRun?: Record<string, ToolProgress[]>;
  skillsByRun?: Record<string, ActivatedSkillBadge[]>;
  artifactsByRun?: Record<string, ArtifactSummary[]>;
  citationsByRun?: Record<string, StoredCitation[]>;
  live?: LiveTurn;
  submittingError?: string;
  submittingErrorCode?: string;
  version: number;
  loadedVersion?: number;
};

const statusLabels: Record<RunState, string> = {
  running: '生成中',
  awaiting_approval: '待审批',
  cancelling: '取消中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
};

const MCP_MANAGEABLE_ERROR_CODES = new Set([
  'MCP_SERVER_UNAVAILABLE',
  'MCP_PERMISSION_DENIED',
  'MCP_TOOL_TIMEOUT',
  'MCP_TOOL_RETURNED_ERROR',
  'MCP_PROTOCOL_ERROR',
  'MCP_RESPONSE_TOO_LARGE',
  'MCP_MALFORMED_RESPONSE',
  'MCP_SERVER_BUSY',
  'MCP_CALL_ABORTED',
  'MCP_TOOL_LIMIT_EXCEEDED',
]);

function isMcpManageableError(code?: string | null): boolean {
  return Boolean(code && MCP_MANAGEABLE_ERROR_CODES.has(code));
}

function failureMessage(code: string | null) {
  switch (code) {
    case 'MODEL_TIMEOUT': return 'DeepSeek 响应超时。';
    case 'MODEL_UPSTREAM_ERROR': return 'DeepSeek 调用失败。';
    case 'RUN_INTERRUPTED': return '后端运行已中断，本次生成已结束。';
    case 'RATE_LIMITED': return '本轮工具调用次数或返回内容已达到上限。';
    case 'RUN_TIMEOUT': return '本轮对话执行总时长已超过上限（300 秒）。';
    case 'MCP_SERVER_UNAVAILABLE': return 'MCP 服务当前未连接或不可用，请检查 MCP 服务连接状态。';
    case 'MCP_PERMISSION_DENIED': return 'MCP 工具未获得有效授权，或服务配置/工具定义已变更，请在 MCP 服务页重新核对授权。';
    case 'MCP_TOOL_TIMEOUT': return 'MCP 工具调用超时，外部执行结果未知。';
    case 'MCP_TOOL_RETURNED_ERROR': return 'MCP 工具返回了错误结果。';
    case 'MCP_PROTOCOL_ERROR': return 'MCP 协议通信失败或连接中断，请检查服务诊断信息。';
    case 'MCP_RESPONSE_TOO_LARGE': return 'MCP 响应体积超过安全上限，已中止读取。';
    case 'MCP_MALFORMED_RESPONSE': return 'MCP 服务返回了畸形或无法解析的数据。';
    case 'MCP_SERVER_BUSY': return '该 MCP 服务当前请求排队已满，请稍后重试。';
    case 'MCP_CALL_ABORTED': return '同轮前置工具调用已失败，已中止后续外部 MCP 调用。';
    case 'MCP_TOOL_LIMIT_EXCEEDED': return '本轮授权工具数量或 Schema 大小超出上限，请前往 MCP 服务页精简授权。';
    case 'MCP_EGRESS_AUTHORIZATION_REQUIRED': return '当前对话包含受保护的本地上下文，需先完成本轮外部数据外发确认。';
    default: return '无法完成本次生成。';
  }
}

function formatDuration(durationMs?: number): string | undefined {
  if (typeof durationMs !== 'number' || Number.isNaN(durationMs) || durationMs < 0) return undefined;
  if (durationMs < 1000) return `${Math.round(durationMs)}ms`;
  return `${(durationMs / 1000).toFixed(1)}s`;
}

function readInitialMcpEnabled(): boolean {
  try {
    return localStorage.getItem(MCP_ENABLED_STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

function readInitialSkillsEnabled(): boolean {
  try {
    return localStorage.getItem(SKILLS_ENABLED_STORAGE_KEY) !== 'false';
  } catch {
    return true;
  }
}

function readInitialSkillsMode(): 'auto' | 'manual' {
  try {
    return localStorage.getItem(SKILLS_MODE_STORAGE_KEY) === 'manual' ? 'manual' : 'auto';
  } catch {
    return 'auto';
  }
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function isPreviewableImage(mediaType: string): boolean {
  return ['image/png', 'image/jpeg', 'image/svg+xml', 'image/webp', 'image/gif'].includes(
    mediaType.toLowerCase().split(';')[0].trim(),
  );
}

function fileIcon(filename: string, mediaType = ''): string {
  const lower = filename.toLowerCase();
  if (lower.endsWith('.docx') || lower.endsWith('.doc')) return '📝';
  if (lower.endsWith('.xlsx') || lower.endsWith('.xls') || lower.endsWith('.csv')) return '📊';
  if (lower.endsWith('.pdf')) return '📑';
  if (lower.endsWith('.pptx')) return '📽️';
  if (lower.endsWith('.mmd') || lower.endsWith('.mermaid')) return '🗺️';
  if (isPreviewableImage(mediaType) || /\.(png|jpe?g|svg|webp|gif)$/.test(lower)) return '🖼️';
  if (lower.endsWith('.md') || lower.endsWith('.txt')) return '📄';
  return '📦';
}

function fileTypeLabel(filename: string, mediaType = ''): string {
  const lower = filename.toLowerCase();
  const ext = lower.includes('.') ? lower.slice(lower.lastIndexOf('.') + 1).toUpperCase() : '';
  if (ext === 'DOCX') return 'Word 文档';
  if (ext === 'XLSX') return 'Excel 表格';
  if (ext === 'PDF') return 'PDF 文档';
  if (ext === 'PPTX') return '演示文稿';
  if (ext === 'MMD') return 'Mermaid 导图';
  if (ext === 'SVG') return 'SVG 矢量图';
  if (ext === 'PNG' || ext === 'JPG' || ext === 'JPEG') return `${ext} 图片`;
  if (ext === 'CSV') return 'CSV 数据';
  if (ext === 'MD') return 'Markdown';
  if (ext) return ext;
  return mediaType || '文件';
}

function inspectHistoricalContextRisk(turns?: Turn[]) {
  let historicalRagRisk = false;
  let historicalFileRisk = false;
  if (!turns?.length) return { historicalRagRisk, historicalFileRisk };
  for (const turn of turns) {
    const run = turn.run;
    if (!run || run.status !== 'completed') continue;
    if (run.citations && run.citations.length > 0) {
      historicalRagRisk = true;
    }
    if (run.tool_calls) {
      for (const tc of run.tool_calls) {
        if (tc.tool_name === 'search_knowledge') {
          historicalRagRisk = true;
        } else if ((tc.source ?? 'native') !== 'mcp') {
          historicalFileRisk = true;
        }
      }
    }
  }
  return { historicalRagRisk, historicalFileRisk };
}

function toolProgress(previous: ToolProgress[], event: ToolEvent): ToolProgress[] {
  const id = event.data.tool_call_id;
  if (event.type === 'tool_started') {
    if (previous.some((tool) => tool.id === id)) return previous;
    return [
      ...previous,
      {
        id,
        name: event.data.tool_name,
        state: 'running',
        source: event.data.source,
        toolCategory: event.data.tool_category ?? undefined,
        serverId: event.data.server_id ?? undefined,
        serverName: event.data.server_name ?? undefined,
        originalToolName: event.data.original_tool_name ?? undefined,
      },
    ];
  }
  const summary = event.type === 'tool_completed' ? event.data.summary : event.data.message;
  const state = event.type === 'tool_completed' ? 'completed' : event.data.code === 'RUN_CANCELLED' ? 'cancelled' : 'failed';
  const errorCode = event.type === 'tool_failed' ? event.data.code : undefined;
  const recoverable = event.type === 'tool_failed' ? Boolean(event.data.recoverable) : undefined;
  return previous.map((tool) => {
    if (tool.id !== id || tool.state !== 'running') return tool;
    return {
      ...tool,
      state,
      summary,
      errorCode,
      recoverable,
      source: event.data.source ?? tool.source,
      toolCategory: event.data.tool_category ?? tool.toolCategory,
      serverId: event.data.server_id ?? tool.serverId,
      serverName: event.data.server_name ?? tool.serverName,
      originalToolName: event.data.original_tool_name ?? tool.originalToolName,
      durationMs: typeof event.data.duration_ms === 'number' ? event.data.duration_ms : tool.durationMs,
    };
  });
}

function toolCategoryBadge(category?: ToolCategory | null): string | null {
  if (category === 'skill') return 'Skill';
  if (category === 'workspace') return 'Workspace';
  if (category === 'runtime') return 'Runtime';
  return null;
}

function ToolActivity({
  tools,
  skills,
  status,
}: {
  tools?: ToolProgress[];
  skills?: ActivatedSkillBadge[];
  status?: RunState;
}) {
  const hasManyCompleted = Boolean(tools && tools.length >= 3 && status && isTerminal(status));
  const [expanded, setExpanded] = useState<boolean>();
  if (!tools?.length && !skills?.length) return null;
  const isOpen = expanded ?? !hasManyCompleted;
  const recoverableCount = tools?.filter((t) => t.state === 'failed' && t.recoverable).length ?? 0;

  return (
    <div className="run-activity-wrap">
      {skills && skills.length > 0 && (
        <div className="activated-skills-row" aria-label="已激活技能">
          {skills.map((sk) => (
            <span key={sk.skill_id} className="activated-skill-pill">
              🧩 {sk.name}
            </span>
          ))}
        </div>
      )}
      {tools && tools.length > 0 && (
        <>
          {hasManyCompleted && (
            <button
              type="button"
              className="activity-collapse-toggle"
              aria-expanded={isOpen}
              onClick={() => setExpanded(!isOpen)}>
              <span>
                {isOpen ? '▾' : '▸'} 执行过程（{tools.length} 步
                {recoverableCount > 0 ? `，含 ${recoverableCount} 次可恢复修正` : ''}）
              </span>
            </button>
          )}
          {isOpen && (
            <ul className="tool-activity" aria-label="工具执行">
              {tools.map((tool) => {
                const isRecoverableFailure = tool.state === 'failed' && tool.recoverable;
                const stateText = tool.state === 'running'
                  ? (status && isTerminal(status) ? '执行已中断' : '执行中')
                  : isRecoverableFailure
                    ? '可恢复错误 · 继续修正'
                    : { completed: '已完成', failed: '执行失败', cancelled: '已取消' }[tool.state];
                const duration = formatDuration(tool.durationMs);
                if (tool.source === 'mcp') {
                  return (
                    <li key={tool.id} role="status" className="tool-activity-mcp" data-state={tool.state} data-source="mcp">
                      <span className="tool-source-badge">MCP</span>
                      {tool.serverName && <span className="tool-server-name">{tool.serverName}</span>}
                      <span className="tool-name">{tool.originalToolName || tool.name}</span>
                      {' · '}
                      <span className={`tool-state-label state-${tool.state}`}>{stateText}</span>
                      {duration && <span className="tool-duration"> · {duration}</span>}
                      {tool.summary && <span className="tool-summary"> · {tool.summary}</span>}
                      {tool.state === 'failed' && tool.errorCode && (
                        <code className="tool-error-code">{tool.errorCode}</code>
                      )}
                    </li>
                  );
                }
                const catBadge = toolCategoryBadge(tool.toolCategory);
                return (
                  <li
                    key={tool.id}
                    role="status"
                    className={`${isRecoverableFailure ? 'tool-recoverable' : ''} ${catBadge ? `tool-cat-${tool.toolCategory}` : ''}`.trim()}
                    data-state={tool.state}>
                    {catBadge && <span className={`tool-cat-badge cat-${tool.toolCategory}`}>{catBadge}</span>}
                    <span className="tool-name">{tool.name}</span>{' · '}
                    <span className={isRecoverableFailure ? 'tool-recoverable-label' : ''}>{stateText}</span>
                    {duration && <span className="tool-duration"> · {duration}</span>}
                    {tool.summary && <span className="tool-summary"> · {tool.summary}</span>}
                  </li>
                );
              })}
            </ul>
          )}
        </>
      )}
    </div>
  );
}

function ArtifactImagePreview({
  client,
  sessionId,
  artifactId,
  filename,
}: {
  client: ApiClient;
  sessionId: string;
  artifactId: string;
  filename: string;
}) {
  const [blobUrl, setBlobUrl] = useState<string>();
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!client.fetchArtifactBlob) return;
    const controller = new AbortController();
    let objectUrl: string | undefined;
    setFailed(false);
    void client.fetchArtifactBlob(sessionId, artifactId, controller.signal).then((blob) => {
      if (controller.signal.aborted) return;
      objectUrl = URL.createObjectURL(blob);
      setBlobUrl(objectUrl);
    }).catch(() => {
      if (!controller.signal.aborted) setFailed(true);
    });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [client, sessionId, artifactId]);

  if (failed || !blobUrl) return null;
  return (
    <div className="artifact-image-wrap">
      <img src={blobUrl} alt={filename} className="artifact-preview-img" />
    </div>
  );
}

function TurnArtifacts({
  client,
  sessionId,
  artifacts,
  onContinueEdit,
}: {
  client?: ApiClient;
  sessionId: string;
  artifacts?: ArtifactSummary[];
  onContinueEdit?: (artifact: ArtifactSummary) => void;
}) {
  const [actionNotice, setActionNotice] = useState('');
  const [busyId, setBusyId] = useState<string>();

  const visible = artifacts?.filter((a) => a.status !== 'discarded') ?? [];
  if (visible.length === 0) return null;

  async function runAction(artifact: ArtifactSummary, kind: 'open' | 'reveal' | 'export') {
    if (busyId) return;
    setBusyId(artifact.artifact_id);
    setActionNotice('');
    try {
      if (kind === 'open') {
        await openArtifact(sessionId, artifact.artifact_id);
      } else if (kind === 'reveal') {
        await revealArtifactInFinder(sessionId, artifact.artifact_id);
      } else {
        const res = await exportArtifact(sessionId, artifact.artifact_id);
        if (res?.exported) setActionNotice(`已另存为：${res.filename}`);
      }
    } catch (err) {
      setActionNotice(mcpErrorMessage(err, '操作文件产物失败，请确认文件存在。'));
    } finally {
      setBusyId(undefined);
    }
  }

  return (
    <section className="turn-artifacts" aria-label="生成的文件产物">
      <div className="turn-artifacts-header">
        <strong>文件产物（{visible.length}）</strong>
        {actionNotice && <span className="artifact-action-notice">{actionNotice}</span>}
      </div>
      <div className="artifact-card-grid">
        {visible.map((artifact) => {
          const isPublished = artifact.status === 'published';
          const missing = artifact.file_exists === false;
          const isBusy = busyId === artifact.artifact_id;
          return (
            <div key={artifact.artifact_id} className={`artifact-card ${missing ? 'is-missing' : ''}`}>
              <div className="artifact-card-head">
                <span className="artifact-icon" aria-hidden="true">
                  {fileIcon(artifact.filename, artifact.media_type)}
                </span>
                <div className="artifact-info">
                  <div className="artifact-name-row">
                    <strong className="artifact-filename" title={artifact.filename}>{artifact.filename}</strong>
                    <span className="artifact-type-badge">{fileTypeLabel(artifact.filename, artifact.media_type)}</span>
                    {!isPublished && <span className="artifact-status-badge">暂存中</span>}
                    {missing && <span className="artifact-status-badge missing">文件已移除</span>}
                  </div>
                  <span className="artifact-meta">
                    {formatBytes(artifact.size_bytes)}
                    {artifact.title && artifact.title !== artifact.filename ? ` · ${artifact.title}` : ''}
                  </span>
                </div>
                {isPublished && !missing && (
                  <div className="artifact-actions">
                    <button
                      type="button"
                      className="artifact-btn primary"
                      disabled={isBusy}
                      onClick={() => void runAction(artifact, 'open')}>
                      打开
                    </button>
                    <button
                      type="button"
                      className="artifact-btn"
                      disabled={isBusy}
                      onClick={() => void runAction(artifact, 'reveal')}>
                      Finder
                    </button>
                    <button
                      type="button"
                      className="artifact-btn"
                      disabled={isBusy}
                      onClick={() => void runAction(artifact, 'export')}>
                      另存为
                    </button>
                    {onContinueEdit && (
                      <button
                        type="button"
                        className="artifact-btn"
                        disabled={isBusy}
                        onClick={() => onContinueEdit(artifact)}>
                        继续编辑
                      </button>
                    )}
                  </div>
                )}
              </div>
              {client && isPublished && !missing && isPreviewableImage(artifact.media_type) && (
                <ArtifactImagePreview
                  client={client}
                  sessionId={sessionId}
                  artifactId={artifact.artifact_id}
                  filename={artifact.filename}
                />
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function resolveTurnTools(liveTools?: ToolProgress[], savedTools?: ToolCallSummary[]): ToolProgress[] | undefined {
  if (liveTools?.length) return liveTools;
  if (!savedTools?.length) return undefined;
  return savedTools.map((tc) => ({
    id: tc.tool_call_id,
    name: tc.tool_name,
    state: tc.status,
    summary: tc.summary ?? undefined,
    errorCode: tc.error_code ?? undefined,
    source: tc.source,
    toolCategory: tc.tool_category ?? undefined,
    recoverable: tc.recoverable,
    serverId: tc.server_id ?? undefined,
    serverName: tc.server_name ?? undefined,
    originalToolName: tc.original_tool_name ?? undefined,
    durationMs: typeof tc.duration_ms === 'number' ? tc.duration_ms : undefined,
  }));
}

function resolveTurnSkills(
  liveSkills?: ActivatedSkillBadge[],
  savedSkills?: StoredSkillActivation[],
): ActivatedSkillBadge[] | undefined {
  if (liveSkills?.length) return liveSkills;
  if (!savedSkills?.length) return undefined;
  return savedSkills.map((s) => ({ skill_id: s.skill_id, name: s.name_snapshot, revision: s.revision }));
}

function resolveTurnArtifacts(
  liveArtifacts?: ArtifactSummary[],
  savedArtifacts?: StoredArtifact[],
): ArtifactSummary[] | undefined {
  if (savedArtifacts?.length) return savedArtifacts;
  if (liveArtifacts?.length) return liveArtifacts;
  return undefined;
}

function CitationStatusTag({ citation }: { citation: StoredCitation }) {
  if (citation.status === 'source_removed') {
    return <span className="citation-status status-removed">来源教材已移除</span>;
  }
  if (citation.status !== 'resolved') {
    return <span className="citation-status status-removed">引用不可用（{citation.status}）</span>;
  }
  if (citation.historical) {
    return <span className="citation-status status-historical">历史版本引用</span>;
  }
  return <span className="citation-status status-resolved">已核验原件</span>;
}

function TurnCitations({ citations }: { citations?: StoredCitation[] }) {
  const [drawerOpen, setDrawerOpen] = useState<boolean>();
  const [expandedId, setExpandedId] = useState<string>();
  if (!citations?.length) return null;
  const isListOpen = drawerOpen ?? citations.length <= 2;
  return (
    <section className="turn-citations" aria-label="教材引用出处">
      <div className="turn-citations-header">
        <strong>教材引用证据（{citations.length}）</strong>
        <button
          type="button"
          className="citations-collapse-btn"
          aria-expanded={isListOpen}
          onClick={() => setDrawerOpen(!isListOpen)}>
          {isListOpen ? '收起列表' : '展开全部引用'}
        </button>
      </div>
      {!isListOpen && (
        <div className="citation-pill-row">
          {citations.map((item, idx) => (
            <button
              key={`${item.evidence_id}-pill-${idx}`}
              type="button"
              className="citation-quick-pill"
              onClick={() => {
                setDrawerOpen(true);
                setExpandedId(`${item.evidence_id}-${idx}`);
              }}>
              <span className="pill-num">[{idx + 1}]</span>
              <span className="pill-book">{item.textbook_name}</span>
              <span className="pill-loc">{item.location_summary}</span>
            </button>
          ))}
        </div>
      )}
      {isListOpen && (
        <ol className="turn-citations-list">
          {citations.map((item, idx) => {
            const isOpen = expandedId === `${item.evidence_id}-${idx}`;
            return (
              <li key={`${item.evidence_id}-${idx}`} className="turn-citation-item">
                <div className="turn-citation-bar">
                  <span className="citation-ref-num">[引用 {idx + 1}]</span>
                  <strong className="citation-book-title">《<bdi>{item.textbook_name}</bdi>》</strong>
                  {item.section_path.length > 0 && (
                    <span className="citation-section-tag">{item.section_path.join(' / ')}</span>
                  )}
                  <span className="badge badge-location">{item.location_summary}</span>
                  <CitationStatusTag citation={item} />
                  {item.warnings.length > 0 && (
                    <span className="badge badge-warning" title="该页含未映射字形或扫描提示，请核对原件">
                      ⚠️ 公式/版式核验提示
                    </span>
                  )}
                  <button
                    type="button"
                    className="citation-toggle-btn"
                    aria-expanded={isOpen}
                    onClick={() => setExpandedId(isOpen ? undefined : `${item.evidence_id}-${idx}`)}>
                    {isOpen ? '收起原文' : '查看证据原文'}
                  </button>
                </div>
                {isOpen && (
                  <div className="turn-citation-detail">
                    {item.warnings.length > 0 && (
                      <p className="evidence-warnings">
                        ⚠️ 解析质量提示：{item.warnings.map((w) => `${w.code}（${w.message}）`).join('；')}
                        ——涉及数学公式或特殊符号时请务必对照原教材该物理页。
                      </p>
                    )}
                    <pre className="evidence-snippet">{item.snippet}</pre>
                  </div>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

function savedTurns(sessionId: string, messages: Message[], runs: RunSummary[]): Turn[] {
  const summaries = new Map(runs.filter((run) => run.session_id === sessionId).map((run) => [run.run_id, run]));
  const committed = messages.filter((message) => message.session_id === sessionId && message.status === 'committed');
  const answers = new Map(committed.filter((message) => message.role === 'assistant')
    .map((message) => [message.run_id, message]));
  return committed.filter((message) => message.role === 'user').map((user) => {
    const summary = summaries.get(user.run_id);
    const run = summary?.user_message_id === user.message_id ? summary : undefined;
    return { user, run, assistant: run?.status === 'completed' ? answers.get(user.run_id) : undefined };
  });
}

async function loadHistory(api: ApiClient, sessionId: string, signal: AbortSignal) {
  let [messages, runs] = await Promise.all([api.listMessages(sessionId, signal), api.listRuns(sessionId, signal)]);
  // Separate reads can straddle a transaction that admits a run or commits its reply.
  const summaryIds = new Set(runs.map((run) => run.run_id));
  if (messages.some((message) => !summaryIds.has(message.run_id))) {
    runs = await api.listRuns(sessionId, signal);
  }
  const userIds = new Set(messages.filter((message) => message.role === 'user').map((message) => message.message_id));
  const answerIds = new Set(messages.filter((message) => message.role === 'assistant').map((message) => message.run_id));
  if (runs.some((run) => !userIds.has(run.user_message_id)
    || (run.status === 'completed' && !answerIds.has(run.run_id)))) {
    messages = await api.listMessages(sessionId, signal);
  }
  return { turns: savedTurns(sessionId, messages, runs),
    running: runs.find((run) => run.session_id === sessionId && !isTerminal(run.status)) };
}

export default function Chat({
  client,
  modelConfigured,
  preview = false,
  defaultRagEnabled = false,
  autoTitleSessions = false,
  mcpSummary,
  onOpenMcpPanel,
  skills,
  onOpenSkillsPanel,
}: {
  client?: ApiClient;
  modelConfigured: boolean;
  preview?: boolean;
  defaultRagEnabled?: boolean;
  autoTitleSessions?: boolean;
  mcpSummary?: McpSummary | null;
  onOpenMcpPanel?: () => void;
  skills?: SkillView[];
  onOpenSkillsPanel?: () => void;
}) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState<string>();
  const [listAttempt, setListAttempt] = useState(0);
  const [listReady, setListReady] = useState(false);
  const [listError, setListError] = useState('');
  const [historyError, setHistoryError] = useState('');
  const [pollError, setPollError] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState('');
  const [renamingId, setRenamingId] = useState<string>();
  const [renameDraft, setRenameDraft] = useState('');
  const [sessionActionId, setSessionActionId] = useState<string>();
  const requests = useRef(new Set<AbortController>());
  const cancellations = useRef(new Map<string, AbortController>());
  const [stopping, setStopping] = useState<string>();
  const [ragEnabled, setRagEnabled] = useState(defaultRagEnabled);
  const [ragMode, setRagMode] = useState<RetrievalMode>('hybrid');
  const [ragCoverage, setRagCoverage] = useState<CoverageMode>('global');
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [textbooks, setTextbooks] = useState<DocumentSummary[]>([]);
  const [textbooksLoading, setTextbooksLoading] = useState(false);
  const [mcpEnabled, setMcpEnabled] = useState<boolean>(readInitialMcpEnabled);
  const [localMcpSummary, setLocalMcpSummary] = useState<McpSummary>();
  const [grantsBySession, setGrantsBySession] = useState<Record<string, FileGrant[]>>({});
  const [egressPrompt, setEgressPrompt] = useState<EgressPrompt>();
  const [grantsOpen, setGrantsOpen] = useState(false);
  const [scopeOpen, setScopeOpen] = useState(false);
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [workspaceBySession, setWorkspaceBySession] = useState<Record<string, SessionWorkspaceView>>({});
  const [pendingAttachmentsBySession, setPendingAttachmentsBySession] = useState<Record<string, string[]>>({});
  const [referencedArtifactsBySession, setReferencedArtifactsBySession] = useState<Record<string, ArtifactSummary[]>>({});
  const [attachmentBusy, setAttachmentBusy] = useState(false);
  const [attachmentError, setAttachmentError] = useState('');
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const [skillsEnabled, setSkillsEnabled] = useState<boolean>(readInitialSkillsEnabled);
  const [skillsMode, setSkillsMode] = useState<'auto' | 'manual'>(readInitialSkillsMode);
  const [selectedSkillIds, setSelectedSkillIds] = useState<string[]>([]);
  const [skillsPickerOpen, setSkillsPickerOpen] = useState(false);
  const turnsEndRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const activeSessionIdRef = useRef<string | undefined>(undefined);

  const effectiveMcpSummary = mcpSummary === undefined ? localMcpSummary : (mcpSummary ?? undefined);
  const enabledSkills = (skills ?? []).filter((s) => s.enabled);

  const handleGrantsChange = useCallback((sid: string, grants: FileGrant[]) => {
    setGrantsBySession((prev) => ({ ...prev, [sid]: grants }));
  }, []);

  function toggleMcpEnabled(next: boolean) {
    setMcpEnabled(next);
    try {
      localStorage.setItem(MCP_ENABLED_STORAGE_KEY, String(next));
    } catch {
      // Ignore storage quota/access errors.
    }
  }

  function toggleSkillsEnabled(next: boolean) {
    setSkillsEnabled(next);
    if (!next) setSkillsPickerOpen(false);
    try {
      localStorage.setItem(SKILLS_ENABLED_STORAGE_KEY, String(next));
    } catch {
      // Ignore storage quota/access errors.
    }
  }

  function changeSkillsMode(next: 'auto' | 'manual') {
    setSkillsMode(next);
    try {
      localStorage.setItem(SKILLS_MODE_STORAGE_KEY, next);
    } catch {
      // Ignore storage quota/access errors.
    }
  }

  const active = listReady ? conversations.find((session) => session.session_id === selected) : undefined;
  const sessionId = active?.session_id;
  const version = active?.version;
  const trackedRunId = active?.activeRunId;
  const live = active?.live;
  const observeStatus = !live || live.lookupComplete || active?.activeStatus === 'cancelling';
  const loaded = !!active && active.loadedVersion === version;
  const busy = !!trackedRunId || !!live;
  const tooLong = !!active && Array.from(active.input.trim()).length > MESSAGE_LIMIT;

  function updateSession(id: string, update: (session: Conversation) => Conversation) {
    setConversations((previous) => previous.map((session) => session.session_id === id ? update(session) : session));
  }

  function reloadHistory(id: string) {
    updateSession(id, (session) => ({ ...session, version: session.version + 1 }));
  }

  function selectSession(id: string) {
    setSelected(id);
    sessionStorage.setItem(SELECTED_SESSION_KEY, id);
  }

  useEffect(() => {
    const pending = requests.current;
    return () => {
      for (const controller of pending) controller.abort();
      pending.clear();
      cancellations.current.clear();
      setCreating(false);
      setStopping(undefined);
    };
  }, [client]);

  useEffect(() => {
    if (!client || mcpSummary !== undefined) return;
    const controller = new AbortController();
    void client.getMcpSummary(controller.signal).then((summary) => {
      if (!controller.signal.aborted) setLocalMcpSummary(summary);
    }).catch(() => {
      // Never impact ordinary chat if MCP summary endpoint is unavailable.
    });
    return () => controller.abort();
  }, [client, mcpSummary, mcpEnabled]);

  useEffect(() => {
    setEgressPrompt(undefined);
    setAttachmentError('');
    activeSessionIdRef.current = sessionId;
  }, [sessionId]);

  const refreshWorkspace = useCallback(async (sid: string, signal?: AbortSignal) => {
    if (!client?.getSessionWorkspace) return;
    try {
      const ws = await client.getSessionWorkspace(sid, signal ?? new AbortController().signal);
      if (!signal?.aborted) {
        setWorkspaceBySession((prev) => ({ ...prev, [sid]: ws }));
      }
    } catch {
      // Workspace metadata fetch failures must not block ordinary chat.
    }
  }, [client]);

  useEffect(() => {
    if (!client || !sessionId || version === undefined) return;
    const controller = new AbortController();
    void refreshWorkspace(sessionId, controller.signal);
    return () => controller.abort();
  }, [client, sessionId, version, refreshWorkspace]);

  const appendImportedFiles = useCallback((sid: string, imported: WorkspaceFileView[]) => {
    if (!imported.length) return;
    setWorkspaceBySession((prev) => {
      const existing = prev[sid] ?? { session_id: sid, files: [], artifacts: [] };
      const byId = new Map(existing.files.map((f) => [f.file_id, f]));
      for (const f of imported) byId.set(f.file_id, f);
      return { ...prev, [sid]: { ...existing, files: Array.from(byId.values()) } };
    });
    setPendingAttachmentsBySession((prev) => {
      const current = prev[sid] ?? [];
      const next = [...current];
      for (const f of imported) {
        if (!next.includes(f.file_id)) next.push(f.file_id);
      }
      return { ...prev, [sid]: next };
    });
  }, []);

  useEffect(() => {
    if (!client || !isTauri()) return;
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    try {
      void getCurrentWebview().onDragDropEvent((event) => {
        const type = event.payload.type;
        if (type === 'enter' || type === 'over') {
          setIsDraggingFiles(true);
        } else if (type === 'leave') {
          setIsDraggingFiles(false);
        } else if (type === 'drop') {
          setIsDraggingFiles(false);
          const sid = activeSessionIdRef.current;
          const paths = event.payload.paths ?? [];
          if (!sid || paths.length === 0) return;
          setAttachmentBusy(true);
          setAttachmentError('');
          void importDroppedWorkspaceFiles(sid, paths).then((imported) => {
            appendImportedFiles(sid, imported);
          }).catch((err) => {
            setAttachmentError(mcpErrorMessage(err, '拖拽导入附件失败。'));
          }).finally(() => {
            setAttachmentBusy(false);
          });
        }
      }).then((fn) => {
        if (cancelled) fn();
        else unlisten = fn;
      }).catch(() => {});
    } catch {
      // Ignore in non-Tauri or test environments.
    }
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [client, appendImportedFiles]);

  async function handlePickAttachments() {
    if (!sessionId || attachmentBusy) return;
    setAttachmentBusy(true);
    setAttachmentError('');
    try {
      const imported = await selectAndImportWorkspaceFiles(sessionId);
      appendImportedFiles(sessionId, imported);
    } catch (err) {
      setAttachmentError(mcpErrorMessage(err, '导入附件失败，请检查文件是否可读。'));
    } finally {
      setAttachmentBusy(false);
    }
  }

  async function handleDeleteWorkspaceFile(sid: string, fileId: string) {
    setAttachmentError('');
    try {
      await deleteWorkspaceFile(sid, fileId);
      setWorkspaceBySession((prev) => {
        const existing = prev[sid];
        if (!existing) return prev;
        return { ...prev, [sid]: { ...existing, files: existing.files.filter((f) => f.file_id !== fileId) } };
      });
      setPendingAttachmentsBySession((prev) => ({
        ...prev,
        [sid]: (prev[sid] ?? []).filter((id) => id !== fileId),
      }));
    } catch (err) {
      setAttachmentError(mcpErrorMessage(err, '删除工作空间文件失败。'));
    }
  }

  function handleContinueEditArtifact(artifact: ArtifactSummary) {
    if (!sessionId) return;
    setReferencedArtifactsBySession((prev) => {
      const current = prev[sessionId] ?? [];
      if (current.some((item) => item.artifact_id === artifact.artifact_id)) return prev;
      return { ...prev, [sessionId]: [...current, artifact] };
    });
    const input = formRef.current?.querySelector('textarea');
    input?.focus();
  }

  useEffect(() => {
    if (!client || !ragEnabled || !sessionId) return;
    const controller = new AbortController();
    setTextbooksLoading(true);
    void client.listDocuments(controller.signal).then((docs) => {
      if (!controller.signal.aborted) {
        setTextbooks(docs);
        setSelectedDocs((prev) => prev.filter((id) => docs.some((d) => d.document_id === id)));
        setTextbooksLoading(false);
      }
    }).catch(() => {
      if (!controller.signal.aborted) setTextbooksLoading(false);
    });
    return () => controller.abort();
  }, [client, ragEnabled, sessionId]);

  useEffect(() => {
    setListReady(false);
    setListError('');
    if (!client) return;
    const controller = new AbortController();
    void client.listSessions(controller.signal).then((items) => {
      if (controller.signal.aborted) return;
      const remembered = sessionStorage.getItem(SELECTED_SESSION_KEY);
      const initial = items.find((item) => item.session_id === remembered)
        ?? items.find((item) => item.active_run_id) ?? items[0];
      setConversations(items.map((item) => ({
        session_id: item.session_id, title: item.title, input: '', version: 0,
        activeRunId: item.active_run_id ?? undefined, activeStatus: item.active_run_status ?? undefined,
      })));
      if (initial) selectSession(initial.session_id);
      else setSelected(undefined);
      setListReady(true);
    }).catch(() => {
      if (!controller.signal.aborted) setListError('无法加载会话列表。');
    });
    return () => controller.abort();
  }, [client, listAttempt]);

  useEffect(() => {
    setHistoryError('');
    if (!client || !sessionId || version === undefined) return;
    const controller = new AbortController();
    void loadHistory(client, sessionId, controller.signal).then(({ turns, running }) => {
      if (controller.signal.aborted) return;
      updateSession(sessionId, (session) => {
        if (session.version !== version) return session;
        const observed = session.live;
        const foundTerminal = observed?.runId && turns.some(({ run }) =>
          run && run.run_id === observed.runId && isTerminal(run.status));
        return {
          ...session, turns, loadedVersion: version, live: foundTerminal ? undefined : observed,
          activeRunId: running?.run_id ?? session.activeRunId,
          activeStatus: running?.run_id === session.activeRunId && session.activeStatus === 'cancelling'
            ? 'cancelling' : running?.status ?? session.activeStatus,
        };
      });
    }).catch(() => {
      if (!controller.signal.aborted) setHistoryError('无法加载聊天历史。');
    });
    return () => controller.abort();
  }, [client, sessionId, version]);

  useEffect(() => {
    setPollError('');
    if (!client || !sessionId || !trackedRunId || !observeStatus || !loaded) return;
    const api = client, id = sessionId, runId = trackedRunId;
    const controller = new AbortController();
    let next: ReturnType<typeof setTimeout>;
    async function query() {
      const run = await api.getRun(id, runId, controller.signal);
      if (controller.signal.aborted) return;
      if (isTerminal(run.status)) {
        settle(id, runId);
      } else {
        updateSession(id, (session) => session.activeRunId !== runId ? session : {
          ...session,
          activeStatus: session.activeStatus === 'cancelling' ? 'cancelling' : run.status,
        });
        next = setTimeout(check, POLL_INTERVAL_MS);
      }
    }
    function check() {
      void query().catch(() => {
        if (!controller.signal.aborted) setPollError('暂时无法确认运行状态。');
      });
    }
    check();
    return () => {
      controller.abort();
      clearTimeout(next);
    };
  }, [client, sessionId, trackedRunId, observeStatus, loaded]);

  useEffect(() => {
    turnsEndRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' });
  }, [active?.turns?.length, live?.draft, sessionId]);

  function settle(id: string, runId: string) {
    updateSession(id, (session) => ({
      ...session, version: session.version + 1,
      activeRunId: session.activeRunId === runId ? undefined : session.activeRunId,
      activeStatus: session.activeRunId === runId ? undefined : session.activeStatus,
      live: session.live?.runId === runId ? { ...session.live, draft: '', settling: true } : session.live,
    }));
  }

  async function createSession(initialInput = '') {
    if (!client || creating) return;
    const controller = new AbortController();
    requests.current.add(controller);
    setCreating(true);
    setCreateError('');
    try {
      const initialTitle = autoTitleSessions ? '新对话' : `会话 ${conversations.length + 1}`;
      const session = await client.createSession(initialTitle, controller.signal);
      if (controller.signal.aborted) return;
      setConversations((previous) => [{ ...session, input: initialInput, version: 0 }, ...previous]);
      selectSession(session.session_id);
    } catch (error) {
      if (!controller.signal.aborted) {
        setCreateError(error instanceof ApiRequestError ? error.message : '无法创建会话。');
      }
    } finally {
      requests.current.delete(controller);
      if (!controller.signal.aborted) setCreating(false);
    }
  }

  function startRenaming(session: Pick<Conversation, 'session_id' | 'title'>) {
    setRenamingId(session.session_id);
    setRenameDraft(session.title);
    setCreateError('');
  }

  function cancelRenaming() {
    setRenamingId(undefined);
    setRenameDraft('');
  }

  async function commitRename(id: string) {
    if (!client || renamingId !== id) return;
    const nextTitle = renameDraft.trim();
    const current = conversations.find((s) => s.session_id === id);
    setRenamingId(undefined);
    if (!nextTitle || !current || nextTitle === current.title) return;
    const bounded = Array.from(nextTitle).slice(0, 60).join('');
    const previousTitle = current.title;
    updateSession(id, (session) => ({ ...session, title: bounded }));
    setSessionActionId(id);
    const controller = new AbortController();
    requests.current.add(controller);
    try {
      const updated = await client.renameSession(id, bounded, controller.signal);
      if (!controller.signal.aborted) {
        updateSession(id, (session) => ({ ...session, title: updated.title }));
      }
    } catch {
      if (!controller.signal.aborted) {
        updateSession(id, (session) => ({ ...session, title: previousTitle }));
        setCreateError('无法重命名会话。');
      }
    } finally {
      requests.current.delete(controller);
      if (!controller.signal.aborted) setSessionActionId((prev) => prev === id ? undefined : prev);
    }
  }

  async function removeSession(id: string) {
    if (!client || sessionActionId) return;
    const target = conversations.find((s) => s.session_id === id);
    if (!target || target.activeRunId || target.live) return;
    setSessionActionId(id);
    setCreateError('');
    const controller = new AbortController();
    requests.current.add(controller);
    try {
      await client.deleteSession(id, controller.signal);
      if (controller.signal.aborted) return;
      const remaining = conversations.filter((s) => s.session_id !== id);
      setConversations(remaining);
      if (selected === id) {
        if (remaining[0]) selectSession(remaining[0].session_id);
        else {
          setSelected(undefined);
          sessionStorage.removeItem(SELECTED_SESSION_KEY);
        }
      }
    } catch {
      if (!controller.signal.aborted) setCreateError('无法删除会话。');
    } finally {
      requests.current.delete(controller);
      if (!controller.signal.aborted) setSessionActionId((prev) => prev === id ? undefined : prev);
    }
  }

  async function stop() {
    if (!client || !sessionId || !trackedRunId || cancellations.current.has(sessionId)) return;
    const id = sessionId, runId = trackedRunId;
    const controller = new AbortController();
    requests.current.add(controller);
    cancellations.current.set(id, controller);
    setStopping(id);
    try {
      const result = await client.cancelRun(runId, controller.signal);
      if (controller.signal.aborted) return;
      if (isTerminal(result.status)) {
        settle(id, runId);
      } else {
        updateSession(id, (session) => session.activeRunId === runId
          ? { ...session, activeStatus: 'cancelling' } : session);
      }
    } catch {
      if (!controller.signal.aborted) {
        updateSession(id, (session) => session.activeRunId !== runId ? session : {
          ...session,
          live: session.live && { ...session.live, detached: true, lookupComplete: true, draft: '',
            problem: '未收到取消结果，正在查询运行状态。' },
        });
        reloadHistory(id);
      }
    } finally {
      requests.current.delete(controller);
      if (cancellations.current.get(id) === controller) cancellations.current.delete(id);
      if (!controller.signal.aborted) setStopping((previous) => previous === id ? undefined : previous);
    }
  }

  async function executeSend(
    targetSession: Conversation,
    question: string,
    { allowExternalDataEgress, forceMcpEnabled }: { allowExternalDataEgress: boolean; forceMcpEnabled?: boolean },
  ) {
    if (!client || !modelConfigured) return;
    const id = targetSession.session_id;
    const useMcp = forceMcpEnabled ?? mcpEnabled;
    const requestId = crypto.randomUUID();
    const controller = new AbortController();
    requests.current.add(controller);
    const shouldAutoTitle = autoTitleSessions && DEFAULT_SESSION_TITLE.test(targetSession.title);
    const nextAutoTitle = shouldAutoTitle ? deriveSessionTitle(question) : undefined;
    updateSession(id, (session) => ({
      ...session,
      title: nextAutoTitle && nextAutoTitle !== session.title ? nextAutoTitle : session.title,
      input: '',
      live: { requestId, question, draft: '' },
      submittingError: undefined,
      submittingErrorCode: undefined,
    }));
    if (nextAutoTitle && nextAutoTitle !== targetSession.title) {
      const titleController = new AbortController();
      requests.current.add(titleController);
      void client.renameSession(id, nextAutoTitle, titleController.signal)
        .then((updated) => {
          if (!titleController.signal.aborted) {
            updateSession(id, (session) => ({ ...session, title: updated.title }));
          }
        })
        .catch(() => {})
        .finally(() => requests.current.delete(titleController));
    }

    let runId: string | undefined;
    const knowledgeOptions = ragEnabled ? {
      enabled: true,
      document_ids: selectedDocs.length > 0 ? selectedDocs : null,
      retrieval_mode: ragMode,
      coverage_mode: ragCoverage,
      top_k: 6,
    } : undefined;
    const mcpOptions = {
      enabled: useMcp,
      allow_external_data_egress: useMcp ? allowExternalDataEgress : false,
    };
    const pendingAttachments = pendingAttachmentsBySession[id] ?? [];
    const referencedArts = referencedArtifactsBySession[id] ?? [];
    const skillsOptions = {
      enabled: skillsEnabled,
      selected_skill_ids: skillsEnabled && skillsMode === 'manual' ? selectedSkillIds : [],
    };
    if (pendingAttachments.length > 0) {
      setPendingAttachmentsBySession((prev) => ({ ...prev, [id]: [] }));
    }
    if (referencedArts.length > 0) {
      setReferencedArtifactsBySession((prev) => ({ ...prev, [id]: [] }));
    }
    try {
      await client.streamRun(id, requestId, question, controller.signal, (item) => {
        if (controller.signal.aborted) return;
        if (item.type === 'run_started') {
          runId = item.run_id;
          updateSession(id, (session) => session.live?.requestId !== requestId ? session : {
            ...session, activeRunId: item.run_id, activeStatus: 'running',
            live: { ...session.live, runId: item.run_id },
          });
        } else if (item.type === 'message_delta') {
          updateSession(id, (session) => session.live?.requestId !== requestId
            || session.live.detached || session.live.settling ? session : {
              ...session, live: { ...session.live, draft: session.live.draft + item.data.delta },
            });
        } else if (item.type === 'tool_started' || item.type === 'tool_completed' || item.type === 'tool_failed') {
          updateSession(id, (session) => {
            if (session.live?.requestId !== requestId || session.live.runId !== item.run_id || session.live.detached) {
              return session;
            }
            const items = toolProgress(session.toolsByRun?.[item.run_id] ?? [], item);
            const nextLive = item.type === 'tool_started' && session.live.draft
              ? { ...session.live, draft: '' }
              : session.live;
            return { ...session, live: nextLive, toolsByRun: { ...session.toolsByRun, [item.run_id]: items } };
          });
        } else if (item.type === 'skill_activated') {
          updateSession(id, (session) => {
            if (session.live?.requestId !== requestId || session.live.runId !== item.run_id || session.live.detached) {
              return session;
            }
            const prev = session.skillsByRun?.[item.run_id] ?? [];
            if (prev.some((s) => s.skill_id === item.data.skill_id)) return session;
            return {
              ...session,
              skillsByRun: {
                ...session.skillsByRun,
                [item.run_id]: [...prev, { skill_id: item.data.skill_id, name: item.data.name, revision: item.data.revision }],
              },
            };
          });
        } else if (item.type === 'artifact_created') {
          updateSession(id, (session) => {
            if (session.live?.requestId !== requestId || session.live.runId !== item.run_id || session.live.detached) {
              return session;
            }
            const prev = session.artifactsByRun?.[item.run_id] ?? [];
            const next = prev.some((a) => a.artifact_id === item.data.artifact_id)
              ? prev.map((a) => (a.artifact_id === item.data.artifact_id ? item.data : a))
              : [...prev, item.data];
            return { ...session, artifactsByRun: { ...session.artifactsByRun, [item.run_id]: next } };
          });
        } else if (item.type === 'run_citations') {
          updateSession(id, (session) => {
            if (session.live?.requestId !== requestId || session.live.runId !== item.run_id || session.live.detached) {
              return session;
            }
            return {
              ...session,
              citationsByRun: { ...session.citationsByRun, [item.run_id]: item.data.citations },
            };
          });
        } else {
          if (item.type === 'run_completed' && item.data.artifacts?.length) {
            updateSession(id, (session) => ({
              ...session,
              artifactsByRun: { ...session.artifactsByRun, [item.run_id]: item.data.artifacts! },
            }));
          }
          settle(id, item.run_id);
        }
      },
      knowledgeOptions,
      mcpOptions,
      skillsOptions,
      pendingAttachments.length > 0 ? pendingAttachments : undefined,
      referencedArts.length > 0 ? referencedArts.map((a) => a.artifact_id) : undefined,
      );
    } catch (error) {
      if (controller.signal.aborted) return;
      if (error instanceof ApiRequestError && !['DUPLICATE_CLIENT_REQUEST', 'RUN_ALREADY_ACTIVE'].includes(error.code ?? '')) {
        const displayMsg = error.code ? mcpErrorMessage(error, error.message) : error.message;
        updateSession(id, (session) => ({
          ...session,
          live: undefined,
          input: session.input || question,
          submittingError: displayMsg,
          submittingErrorCode: error.code,
        }));
        if (error.code === 'MCP_EGRESS_AUTHORIZATION_REQUIRED') {
          const hist = inspectHistoricalContextRisk(targetSession.turns);
          const activeGrants = grantsBySession[id] ?? [];
          const hasLocalCurrent = ragEnabled || activeGrants.length > 0;
          setEgressPrompt({
            sessionId: id,
            question,
            currentRagRisk: ragEnabled,
            currentFileGrantRisk: activeGrants.length > 0,
            historicalRagRisk: hist.historicalRagRisk,
            historicalFileRisk: hist.historicalFileRisk,
            backendMessage: error.message,
            backendForcedHistoryRisk: !hasLocalCurrent || hist.historicalRagRisk || hist.historicalFileRisk,
          });
        }
        return;
      }
      updateSession(id, (session) => session.live?.requestId !== requestId ? session : {
        ...session, live: { ...session.live, draft: '', detached: true },
      });
      try {
        if (error instanceof ApiRequestError && error.code === 'RUN_ALREADY_ACTIVE') {
          updateSession(id, (session) => ({
            ...session, live: undefined, submittingError: error.message, submittingErrorCode: error.code,
          }));
          reloadHistory(id);
          return;
        }
        const found = await client.lookupRun(id, requestId, controller.signal);
        if (controller.signal.aborted) return;
        if (runId && found.run_id !== runId) throw new Error('运行标识不符。');
        updateSession(id, (session) => session.live?.requestId !== requestId ? session : {
          ...session, activeRunId: found.run_id,
          live: { ...session.live, runId: found.run_id, lookupComplete: true },
        });
      } catch {
        if (controller.signal.aborted) return;
        if (runId) {
          updateSession(id, (session) => ({
            ...session, submittingError: '暂时无法确认运行状态。', submittingErrorCode: undefined,
            live: session.live && { ...session.live, lookupComplete: true },
          }));
        } else {
          updateSession(id, (session) => ({
            ...session, live: undefined, input: session.input || question,
            submittingError: '连接中断，暂时无法确认请求是否已提交。', submittingErrorCode: undefined,
          }));
          reloadHistory(id);
        }
      }
    } finally {
      requests.current.delete(controller);
    }
  }

  async function send(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!client || !active || !modelConfigured || busy || !loaded || tooLong) return;
    const question = active.input.trim();
    if (!question) return;

    const hasAvailableMcpTools = effectiveMcpSummary === undefined || effectiveMcpSummary.effective_allowed_tools > 0;
    if (mcpEnabled && hasAvailableMcpTools) {
      const activeGrants = grantsBySession[active.session_id] ?? [];
      const ws = workspaceBySession[active.session_id];
      const hasWorkspaceFiles = (pendingAttachmentsBySession[active.session_id]?.length ?? 0) > 0
        || (ws?.files.length ?? 0) > 0;
      const currentRagRisk = ragEnabled;
      const currentFileGrantRisk = activeGrants.length > 0 || hasWorkspaceFiles;
      const { historicalRagRisk, historicalFileRisk } = inspectHistoricalContextRisk(active.turns);
      if (currentRagRisk || currentFileGrantRisk || historicalRagRisk || historicalFileRisk) {
        setEgressPrompt({
          sessionId: active.session_id,
          question,
          currentRagRisk,
          currentFileGrantRisk,
          historicalRagRisk,
          historicalFileRisk,
        });
        return;
      }
    }

    await executeSend(active, question, { allowExternalDataEgress: false });
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (client && active && modelConfigured && !busy && loaded && !tooLong && active.input.trim()) {
        formRef.current?.requestSubmit();
      }
    }
  }

  const showPreview = preview && !client;
  if (!client && !showPreview) return null;
  const liveSaved = !!live?.runId && active?.turns?.some(({ user }) => user.run_id === live.runId);
  const showDraft = live?.draft && active?.activeStatus === 'running' && !live.settling;
  const canSend = !!active && loaded && modelConfigured && !busy && !!active.input.trim() && !tooLong;

  return (
    <section className="chat" aria-label="聊天">
      <aside className="session-sidebar">
        <div className="session-sidebar-top">
          <button
            type="button"
            className="new-chat-btn"
            onClick={() => void createSession()}
            disabled={!client || !listReady || creating}>
            {creating ? '创建中' : '新建会话'}
          </button>
          {active && (
            <select
              className="sr-only"
              aria-label="当前会话"
              value={selected}
              onChange={(event) => selectSession(event.target.value)}>
              {conversations.map((session) => (
                <option key={session.session_id} value={session.session_id}>{session.title}</option>
              ))}
            </select>
          )}
        </div>

        <div className="session-list-label">历史会话</div>

        {listError && (
          <p role="alert" className="sidebar-msg">
            {listError}{' '}
            <button type="button" onClick={() => setListAttempt((attempt) => attempt + 1)}>重新加载会话</button>
          </p>
        )}
        {client && !listReady && !listError && <p role="status" className="sidebar-msg">正在加载会话</p>}
        {createError && <p role="alert" className="sidebar-msg">{createError}</p>}

        <div className="session-list" role="list">
          {conversations.map((session) => {
            const isActive = session.session_id === selected;
            const isRunning = !!session.activeRunId || !!session.live;
            const isEditing = renamingId === session.session_id;
            return (
              <div key={session.session_id} className={`session-row ${isActive ? 'active' : ''}`} role="listitem">
                {isEditing ? (
                  <input
                    type="text"
                    className="session-rename-input"
                    aria-label="编辑会话标题"
                    value={renameDraft}
                    maxLength={60}
                    autoFocus
                    onChange={(e) => setRenameDraft(e.target.value)}
                    onBlur={() => void commitRename(session.session_id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.nativeEvent.isComposing) {
                        e.preventDefault();
                        void commitRename(session.session_id);
                      } else if (e.key === 'Escape') {
                        e.preventDefault();
                        cancelRenaming();
                      }
                    }}
                  />
                ) : (
                  <>
                    <button
                      type="button"
                      className={`session-list-item ${isActive ? 'active' : ''}`}
                      onClick={() => selectSession(session.session_id)}
                      onDoubleClick={() => startRenaming(session)}
                      title={session.title}>
                      <span className="session-item-icon" aria-hidden="true">💬</span>
                      <span className="session-item-title">{session.title}</span>
                      {isRunning && <span className="session-running-dot" title="生成中" />}
                    </button>
                    <div className="session-row-actions">
                      <button
                        type="button"
                        className="session-mini-btn"
                        aria-label={`重命名 ${session.title}`}
                        title="重命名"
                        disabled={sessionActionId === session.session_id}
                        onClick={(e) => {
                          e.stopPropagation();
                          startRenaming(session);
                        }}>
                        ✎
                      </button>
                      <button
                        type="button"
                        className="session-mini-btn danger"
                        aria-label={`删除 ${session.title}`}
                        title={isRunning ? '生成中无法删除' : '删除会话'}
                        disabled={isRunning || sessionActionId === session.session_id}
                        onClick={(e) => {
                          e.stopPropagation();
                          void removeSession(session.session_id);
                        }}>
                        ✕
                      </button>
                    </div>
                  </>
                )}
              </div>
            );
          })}
          {listReady && conversations.length === 0 && !listError && (
            <p className="session-empty-hint">点击上方“新建会话”开始提问</p>
          )}
        </div>
      </aside>

      <div className={`chat-stage ${isDraggingFiles ? 'is-dragging' : ''}`}>
        {isDraggingFiles && active && (
          <div className="chat-drop-overlay" aria-live="polite">
            <div className="chat-drop-card">
              <span>📂</span>
              <strong>松开鼠标，将文件导入当前会话工作空间</strong>
              <small>原始文件保留不动，仅向工作空间复制一份副本</small>
            </div>
          </div>
        )}
        {showPreview && (
          <div className="composer-dock">
            <div className="composer">
              <label htmlFor="question" className="sr-only">问题</label>
              <textarea id="question" rows={3} disabled placeholder="请在 KnowledgeAgent 桌面窗口中开始对话…" />
              <div className="composer-actions">
                <button type="button" disabled>发送</button>
              </div>
            </div>
          </div>
        )}

        {client && !active && listReady && (
          <div className="chat-no-session">
            <div className="welcome-emblem" aria-hidden="true">KA</div>
            <h2>开始一段新的学术对话</h2>
            <p>点击左侧「新建会话」，即可结合您的本地教材库进行 RAG 溯源问答与公式推导。</p>
            <button type="button" onClick={() => void createSession()} disabled={creating}>
              {creating ? '创建中' : '立即新建会话'}
            </button>
          </div>
        )}

        {client && active && (
          <>
            <header className="chat-topbar">
              <div className="chat-topbar-left">
                <h2
                  className="chat-session-title"
                  title="双击重命名会话"
                  onDoubleClick={() => startRenaming(active)}>
                  {active.title}
                </h2>
                <button
                  type="button"
                  className="topbar-rename-btn"
                  aria-label="重命名当前会话"
                  title="重命名当前会话"
                  onClick={() => startRenaming(active)}>
                  ✎
                </button>
                <span className="chat-model-badge">KnowledgeAgent 学术助教</span>
              </div>
              <div className="chat-topbar-right">
                {(() => {
                  const ws = workspaceBySession[active.session_id];
                  const wsCount = (ws?.files.length ?? 0) + (ws?.artifacts.filter((a) => a.status === 'published').length ?? 0);
                  return (
                    <button
                      type="button"
                      className={`topbar-chip ${workspaceOpen ? 'active' : ''}`}
                      aria-expanded={workspaceOpen}
                      onClick={() => {
                        setWorkspaceOpen((prev) => !prev);
                        setGrantsOpen(false);
                        setScopeOpen(false);
                      }}>
                      <span aria-hidden="true">📂</span>
                      <span>工作空间{wsCount > 0 ? ` (${wsCount})` : ''}</span>
                    </button>
                  );
                })()}
                {ragEnabled && (
                  <button
                    type="button"
                    className={`topbar-chip ${scopeOpen ? 'active' : ''}`}
                    aria-expanded={scopeOpen}
                    onClick={() => {
                      setScopeOpen((prev) => !prev);
                      setGrantsOpen(false);
                      setWorkspaceOpen(false);
                    }}>
                    <span aria-hidden="true">📚</span>
                    <span>
                      {selectedDocs.length > 0
                        ? `已选 ${selectedDocs.length} 本教材`
                        : textbooks.length > 0
                          ? `全部教材 (${textbooks.length})`
                          : '教材范围'}
                    </span>
                  </button>
                )}
                <button
                  type="button"
                  className={`topbar-chip ${grantsOpen ? 'active' : ''}`}
                  aria-expanded={grantsOpen}
                  onClick={() => {
                    setGrantsOpen((prev) => !prev);
                    setScopeOpen(false);
                    setWorkspaceOpen(false);
                  }}>
                  <span aria-hidden="true">📎</span>
                  <span>文件授权</span>
                </button>
              </div>
            </header>

            <div className={`context-drawer ${workspaceOpen ? 'is-open' : 'is-closed'}`}>
              {(() => {
                const ws = workspaceBySession[active.session_id];
                const files = ws?.files ?? [];
                const artifacts = (ws?.artifacts ?? []).filter((a) => a.status === 'published');
                const pendingIds = pendingAttachmentsBySession[active.session_id] ?? [];
                return (
                  <div className="workspace-drawer" aria-label="会话工作空间">
                    <div className="workspace-drawer-head">
                      <strong>当前会话工作空间</strong>
                      <button
                        type="button"
                        className="artifact-btn primary"
                        disabled={attachmentBusy}
                        onClick={() => void handlePickAttachments()}>
                        {attachmentBusy ? '导入中…' : '+ 添加文件'}
                      </button>
                    </div>
                    <div className="workspace-drawer-grid">
                      <div>
                        <h4>已导入附件（{files.length}）</h4>
                        {files.length === 0 ? (
                          <p className="workspace-empty">可将 Word、Markdown、表格、图片等拖拽进聊天窗口或点击“添加文件”。</p>
                        ) : (
                          <ul className="workspace-file-list">
                            {files.map((file) => {
                              const attached = pendingIds.includes(file.file_id);
                              return (
                                <li key={file.file_id}>
                                  <span>{fileIcon(file.original_name, file.media_type)}</span>
                                  <span className="workspace-file-name" title={file.original_name}>{file.original_name}</span>
                                  <span className="workspace-file-meta">{formatBytes(file.size_bytes)}</span>
                                  <button
                                    type="button"
                                    className={`artifact-btn ${attached ? 'primary' : ''}`}
                                    onClick={() => setPendingAttachmentsBySession((prev) => {
                                      const cur = prev[active.session_id] ?? [];
                                      return {
                                        ...prev,
                                        [active.session_id]: attached
                                          ? cur.filter((id) => id !== file.file_id)
                                          : [...cur, file.file_id],
                                      };
                                    })}>
                                    {attached ? '已附带' : '本轮引用'}
                                  </button>
                                  <button
                                    type="button"
                                    className="artifact-btn danger"
                                    onClick={() => void handleDeleteWorkspaceFile(active.session_id, file.file_id)}>
                                    删除
                                  </button>
                                </li>
                              );
                            })}
                          </ul>
                        )}
                      </div>
                      <div>
                        <h4>历史产物（{artifacts.length}）</h4>
                        {artifacts.length === 0 ? (
                          <p className="workspace-empty">本会话生成并交付的 Word、思维导图等文件会保存在这里。</p>
                        ) : (
                          <ul className="workspace-file-list">
                            {artifacts.map((art) => (
                              <li key={art.artifact_id}>
                                <span>{fileIcon(art.filename, art.media_type)}</span>
                                <span className="workspace-file-name" title={art.filename}>{art.filename}</span>
                                <span className="workspace-file-meta">{formatBytes(art.size_bytes)}</span>
                                <button
                                  type="button"
                                  className="artifact-btn"
                                  onClick={() => void openArtifact(active.session_id, art.artifact_id)}>
                                  打开
                                </button>
                                <button
                                  type="button"
                                  className="artifact-btn"
                                  onClick={() => handleContinueEditArtifact(art)}>
                                  继续编辑
                                </button>
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>

            <div className={`context-drawer ${grantsOpen ? 'is-open' : 'is-closed'}`}>
              <GrantPanel client={client} sessionId={active.session_id} onGrantsChange={handleGrantsChange} />
            </div>

            {ragEnabled && (
              <div className={`context-drawer scope-drawer ${scopeOpen ? 'is-open' : 'is-closed'}`}>
                <div className="knowledge-scope-selector">
                  {textbooksLoading ? (
                    <small role="status">正在读取已导入教材列表…</small>
                  ) : textbooks.length === 0 ? (
                    <small>当前尚未导入教材。可前往左侧「教材知识库」导入并建立索引。</small>
                  ) : (
                    <div className="library-doc-filters" role="group" aria-label="选择问答教材范围">
                      <span className="filter-label">检索教材范围：</span>
                      <button
                        type="button"
                        className={`filter-chip ${selectedDocs.length === 0 ? 'active' : ''}`}
                        onClick={() => setSelectedDocs([])}>
                        全部教材（{textbooks.length}）
                      </button>
                      {textbooks.map((doc) => {
                        const isSelected = selectedDocs.includes(doc.document_id);
                        return (
                          <button
                            key={doc.document_id}
                            type="button"
                            className={`filter-chip ${isSelected ? 'active' : ''}`}
                            aria-pressed={isSelected}
                            onClick={() => setSelectedDocs((prev) =>
                              prev.includes(doc.document_id)
                                ? prev.filter((id) => id !== doc.document_id)
                                : [...prev, doc.document_id])}>
                            <bdi>{doc.name}</bdi>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="turns" aria-label="聊天消息">
              <div className="turns-inner">
                {!loaded && !historyError && <p role="status" className="turns-status-msg">正在读取聊天记录</p>}
                {historyError && (
                  <p role="alert" className="turns-status-msg">
                    {historyError}{' '}
                    <button type="button" onClick={() => reloadHistory(active.session_id)}>重新加载记录</button>
                  </p>
                )}

                {loaded && !historyError && (!active.turns || active.turns.length === 0) && !live && (
                  <div className="chat-empty-state">
                    <div className="welcome-emblem" aria-hidden="true">KA</div>
                    <h3>有什么学术问题我可以帮您？</h3>
                    <p>
                      我是 <strong>KnowledgeAgent</strong>，会自动检索您的本地教材库，给出带<strong>章节页码引用</strong>的严谨推导与标准数学公式排版。
                    </p>
                    <div className="prompt-suggestions">
                      {[
                        '讲解一下最大似然估计的原理与推导步骤',
                        '讲解一下拟合优度检验与卡方检验统计量',
                        '对比不同教材中关于勒贝格积分的定义与性质',
                        '数值分析中牛顿迭代法的收敛阶如何证明？',
                      ].map((promptText) => (
                        <button
                          key={promptText}
                          type="button"
                          className="prompt-suggestion-chip"
                          onClick={() => updateSession(active.session_id, (s) => ({ ...s, input: promptText }))}>
                          <span className="suggestion-icon" aria-hidden="true">✦</span>
                          <span>{promptText}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {active.turns?.map(({ user, run, assistant }) => {
                  const status = user.run_id === trackedRunId ? active.activeStatus ?? run?.status : run?.status;
                  const turnTools = resolveTurnTools(active.toolsByRun?.[user.run_id], run?.tool_calls);
                  const turnSkills = resolveTurnSkills(active.skillsByRun?.[user.run_id], run?.skill_activations);
                  const turnArtifacts = resolveTurnArtifacts(active.artifactsByRun?.[user.run_id], run?.artifacts);
                  const turnCitations = active.citationsByRun?.[user.run_id] ?? run?.citations;
                  return (
                    <article key={user.message_id} className="turn">
                      <div className="turn-user-row">
                        <div className="user-bubble">
                          <p className="question">{user.content}</p>
                        </div>
                      </div>
                      <div className="turn-assistant-row">
                        <div className="assistant-avatar" aria-hidden="true">KA</div>
                        <div className="assistant-body">
                          <div className="assistant-meta">
                            <span className="assistant-name">KnowledgeAgent</span>
                            <small role="status" data-status={status}>
                              {status ? statusLabels[status] : '运行状态未知'}
                            </small>
                          </div>
                          <ToolActivity tools={turnTools} skills={turnSkills} status={status} />
                          {assistant && (
                            <MarkdownAnswer
                              ariaLabel="已保存回答"
                              messageId={assistant.message_id}
                              content={assistant.content}
                            />
                          )}
                          {status === 'failed' && run && (
                            <p role="alert" className="turn-error-banner">
                              <span>
                                {run.error_message || failureMessage(run.error_code)} {run.error_code && <code>{run.error_code}</code>}
                              </span>
                              {onOpenMcpPanel && isMcpManageableError(run.error_code) && (
                                <button type="button" className="inline-jump-btn" onClick={onOpenMcpPanel}>
                                  前往 MCP 服务
                                </button>
                              )}
                            </p>
                          )}
                          {user.run_id === live?.runId && showDraft && (
                            <MarkdownAnswer ariaLabel="回答草稿" content={live.draft} />
                          )}
                          <TurnArtifacts
                            client={client}
                            sessionId={active.session_id}
                            artifacts={turnArtifacts}
                            onContinueEdit={handleContinueEditArtifact}
                          />
                          <TurnCitations citations={turnCitations} />
                        </div>
                      </div>
                    </article>
                  );
                })}

                {live && !liveSaved && (
                  <article className="turn turn-live">
                    <div className="turn-user-row">
                      <div className="user-bubble">
                        <p className="question">{live.question}</p>
                      </div>
                    </div>
                    <div className="turn-assistant-row">
                      <div className="assistant-avatar" aria-hidden="true">KA</div>
                      <div className="assistant-body">
                        <div className="assistant-meta">
                          <span className="assistant-name">KnowledgeAgent</span>
                          <small role="status" data-status={active.activeStatus ?? 'running'}>
                            {live.settling ? '正在读取运行结果'
                              : active.activeStatus ? statusLabels[active.activeStatus]
                                : live.detached ? '运行状态未知' : '发送中'}
                          </small>
                        </div>
                        {live.runId && (
                          <ToolActivity
                            tools={active.toolsByRun?.[live.runId]}
                            skills={active.skillsByRun?.[live.runId]}
                            status={active.activeStatus}
                          />
                        )}
                        {showDraft && <MarkdownAnswer ariaLabel="回答草稿" content={live.draft} />}
                        {live.runId && (
                          <TurnArtifacts
                            client={client}
                            sessionId={active.session_id}
                            artifacts={active.artifactsByRun?.[live.runId]}
                            onContinueEdit={handleContinueEditArtifact}
                          />
                        )}
                        {live.runId && <TurnCitations citations={active.citationsByRun?.[live.runId]} />}
                      </div>
                    </div>
                  </article>
                )}

                {live?.detached && <p className="turns-status-msg">连接中断，实时回答已停止接收。</p>}
                {live?.problem && <p role="alert" className="turns-status-msg">{live.problem}</p>}
                {pollError && (
                  <p role="alert" className="turns-status-msg">
                    {pollError}{' '}
                    <button type="button" onClick={() => reloadHistory(active.session_id)}>查询状态</button>
                  </p>
                )}
                <div ref={turnsEndRef} />
              </div>
            </div>

            <div className="composer-dock">
              {mcpEnabled && effectiveMcpSummary?.exceeds_tool_limit && (
                <div role="alert" className="composer-mcp-warning">
                  <span>
                    当前有效授权 MCP 工具数（{effectiveMcpSummary.effective_allowed_tools}）超过单轮上限（{effectiveMcpSummary.max_tools_per_run}），发送前请减少授权工具。
                  </span>
                  {onOpenMcpPanel && (
                    <button type="button" className="inline-jump-btn" onClick={onOpenMcpPanel}>
                      管理 MCP 工具
                    </button>
                  )}
                </div>
              )}

              {egressPrompt && egressPrompt.sessionId === active.session_id && (() => {
                const hasHistoricalRisk = egressPrompt.historicalRagRisk
                  || egressPrompt.historicalFileRisk
                  || Boolean(egressPrompt.backendForcedHistoryRisk);
                const canEliminateByTurningOffRag = !hasHistoricalRisk
                  && egressPrompt.currentRagRisk
                  && !egressPrompt.currentFileGrantRisk;
                const canEliminateByManagingGrants = !hasHistoricalRisk && egressPrompt.currentFileGrantRisk;
                const reasons: string[] = [];
                if (egressPrompt.currentRagRisk) reasons.push('本轮已开启教材知识库 RAG');
                if (egressPrompt.currentFileGrantRisk) reasons.push('当前会话存在本地文件读取授权');
                if (egressPrompt.historicalRagRisk) reasons.push('会话历史包含教材检索上下文');
                if (egressPrompt.historicalFileRisk) reasons.push('会话历史包含已读取的本地文件内容');
                if (egressPrompt.backendForcedHistoryRisk && !egressPrompt.historicalRagRisk && !egressPrompt.historicalFileRisk) {
                  reasons.push('会话包含旧版本或来源不可验证的历史对话上下文');
                }
                return (
                  <div className="egress-confirm-card" role="dialog" aria-modal="false" aria-label="本轮外部数据外发确认">
                    <div className="egress-confirm-header">
                      <strong>外部数据外发安全确认（仅本轮有效）</strong>
                    </div>
                    <p className="egress-confirm-body">
                      当前对话可能包含本地文件或教材内容。允许 MCP 工具协作时，模型可能将部分对话内容发送给外部 MCP 服务。
                    </p>
                    {reasons.length > 0 && (
                      <p className="egress-confirm-reasons">
                        触发原因：{reasons.join('；')}。
                        {hasHistoricalRisk && '由于历史上下文已保留在当前对话中，仅关闭当前开关无法消除历史内容外发风险。'}
                      </p>
                    )}
                    {egressPrompt.backendMessage && (
                      <p className="egress-confirm-backend">{egressPrompt.backendMessage}</p>
                    )}
                    <div className="egress-confirm-actions">
                      <button
                        type="button"
                        className="mcp-btn primary sm"
                        onClick={() => {
                          const targetQuestion = egressPrompt.question;
                          setEgressPrompt(undefined);
                          void executeSend(active, targetQuestion, { allowExternalDataEgress: true });
                        }}>
                        仅本轮授权必要的数据外发
                      </button>
                      <button
                        type="button"
                        className="mcp-btn secondary sm"
                        onClick={() => {
                          toggleMcpEnabled(false);
                          setEgressPrompt(undefined);
                        }}>
                        关闭 MCP
                      </button>
                      {canEliminateByTurningOffRag && (
                        <button
                          type="button"
                          className="mcp-btn secondary sm"
                          onClick={() => {
                            setRagEnabled(false);
                            setEgressPrompt(undefined);
                          }}>
                          关闭教材 RAG
                        </button>
                      )}
                      {canEliminateByManagingGrants && (
                        <>
                          {egressPrompt.currentRagRisk && (
                            <button
                              type="button"
                              className="mcp-btn secondary sm"
                              onClick={() => {
                                setRagEnabled(false);
                                setEgressPrompt(undefined);
                              }}>
                              关闭教材 RAG
                            </button>
                          )}
                          <button
                            type="button"
                            className="mcp-btn secondary sm"
                            onClick={() => {
                              setGrantsOpen(true);
                              setScopeOpen(false);
                              setEgressPrompt(undefined);
                            }}>
                            管理文件授权
                          </button>
                        </>
                      )}
                      <button
                        type="button"
                        className="mcp-btn secondary sm"
                        disabled={creating}
                        onClick={() => {
                          const draftQuestion = egressPrompt.question;
                          setEgressPrompt(undefined);
                          void createSession(draftQuestion);
                        }}>
                        开启新对话
                      </button>
                      <button
                        type="button"
                        className="mcp-btn ghost sm"
                        onClick={() => setEgressPrompt(undefined)}>
                        取消
                      </button>
                    </div>
                  </div>
                );
              })()}

              <form ref={formRef} className="composer" onSubmit={send}>
                {(() => {
                  const ws = workspaceBySession[active.session_id];
                  const pendingIds = pendingAttachmentsBySession[active.session_id] ?? [];
                  const pendingFiles = pendingIds
                    .map((fid) => ws?.files.find((f) => f.file_id === fid))
                    .filter((f): f is WorkspaceFileView => Boolean(f));
                  const refArts = referencedArtifactsBySession[active.session_id] ?? [];
                  if (pendingFiles.length === 0 && refArts.length === 0 && !attachmentError) return null;
                  return (
                    <div className="composer-attachments" aria-label="本轮附件与引用产物">
                      {pendingFiles.map((file) => (
                        <span key={file.file_id} className="attachment-chip">
                          <span>{fileIcon(file.original_name, file.media_type)}</span>
                          <span className="attachment-chip-name" title={file.original_name}>{file.original_name}</span>
                          <span className="attachment-chip-size">{formatBytes(file.size_bytes)}</span>
                          <button
                            type="button"
                            className="attachment-chip-remove"
                            aria-label={`移除附件 ${file.original_name}`}
                            onClick={() => setPendingAttachmentsBySession((prev) => ({
                              ...prev,
                              [active.session_id]: (prev[active.session_id] ?? []).filter((id) => id !== file.file_id),
                            }))}>
                            ✕
                          </button>
                        </span>
                      ))}
                      {refArts.map((art) => (
                        <span key={art.artifact_id} className="attachment-chip is-artifact">
                          <span>✏️</span>
                          <span className="attachment-chip-name" title={art.filename}>继续编辑：{art.filename}</span>
                          <button
                            type="button"
                            className="attachment-chip-remove"
                            aria-label={`取消引用产物 ${art.filename}`}
                            onClick={() => setReferencedArtifactsBySession((prev) => ({
                              ...prev,
                              [active.session_id]: (prev[active.session_id] ?? []).filter((a) => a.artifact_id !== art.artifact_id),
                            }))}>
                            ✕
                          </button>
                        </span>
                      ))}
                      {attachmentError && <span role="alert" className="attachment-error-inline">{attachmentError}</span>}
                    </div>
                  );
                })()}

                <label htmlFor="question" className="sr-only">问题</label>
                <textarea
                  id="question"
                  rows={2}
                  value={active.input}
                  placeholder="给 KnowledgeAgent 发送消息，或拖拽文件到此导入工作空间…（Enter 发送，Shift + Enter 换行）"
                  onKeyDown={handleComposerKeyDown}
                  onChange={(event) => {
                    const input = event.target.value;
                    updateSession(active.session_id, (session) => ({ ...session, input }));
                  }}
                />
                {skillsEnabled && skillsPickerOpen && (
                  <div className="skills-picker-bar" role="region" aria-label="选择本轮 Skills">
                    <div className="skills-picker-head">
                      <label className="skills-mode-option">
                        <input
                          type="radio"
                          name="skills-mode"
                          checked={skillsMode === 'auto'}
                          onChange={() => changeSkillsMode('auto')}
                        />
                        <span>自动匹配（推荐）</span>
                      </label>
                      <label className="skills-mode-option">
                        <input
                          type="radio"
                          name="skills-mode"
                          checked={skillsMode === 'manual'}
                          onChange={() => changeSkillsMode('manual')}
                        />
                        <span>手动指定</span>
                      </label>
                      {onOpenSkillsPanel && (
                        <button type="button" className="inline-jump-btn" onClick={onOpenSkillsPanel}>
                          管理 Skills
                        </button>
                      )}
                    </div>
                    {skillsMode === 'manual' && (
                      <div className="skills-picker-chips">
                        {enabledSkills.length === 0 ? (
                          <span className="skills-picker-empty">暂无已启用技能，可前往 Skills 页启用或导入。</span>
                        ) : (
                          enabledSkills.map((sk) => {
                            const selectedSk = selectedSkillIds.includes(sk.skill_id);
                            return (
                              <label key={sk.skill_id} className={`skill-select-chip ${selectedSk ? 'selected' : ''}`}>
                                <input
                                  type="checkbox"
                                  checked={selectedSk}
                                  onChange={() => setSelectedSkillIds((prev) =>
                                    prev.includes(sk.skill_id)
                                      ? prev.filter((id) => id !== sk.skill_id)
                                      : [...prev, sk.skill_id],
                                  )}
                                />
                                <span>{sk.name}</span>
                              </label>
                            );
                          })
                        )}
                      </div>
                    )}
                  </div>
                )}
                <div className="composer-bottom-bar">
                  <section className="knowledge-qa-inline" aria-label="教材知识库与工具设置">
                    <button
                      type="button"
                      className="attach-file-btn"
                      disabled={attachmentBusy || busy}
                      onClick={() => void handlePickAttachments()}
                      title="选择本地文件导入当前会话工作空间">
                      {attachmentBusy ? '导入中…' : '📎 附件'}
                    </button>
                    <label className={`rag-pill-toggle ${ragEnabled ? 'is-on' : ''}`}>
                      <input
                        type="checkbox"
                        aria-label="结合教材知识库"
                        checked={ragEnabled}
                        onChange={(e) => setRagEnabled(e.target.checked)}
                      />
                      <span>教材知识库 RAG</span>
                    </label>
                    <label className={`rag-pill-toggle mcp-pill-toggle ${mcpEnabled ? 'is-on' : ''}`}>
                      <input
                        type="checkbox"
                        aria-label="启用 MCP 工具"
                        checked={mcpEnabled}
                        onChange={(e) => toggleMcpEnabled(e.target.checked)}
                      />
                      <span>
                        MCP 工具
                        {effectiveMcpSummary !== undefined
                          ? `（${effectiveMcpSummary.effective_allowed_tools} 个可用）`
                          : ''}
                      </span>
                    </label>
                    <div className="skills-pill-group">
                      <label className={`rag-pill-toggle skills-pill-toggle ${skillsEnabled ? 'is-on' : ''}`}>
                        <input
                          type="checkbox"
                          aria-label="启用 Skills"
                          checked={skillsEnabled}
                          onChange={(e) => toggleSkillsEnabled(e.target.checked)}
                        />
                        <span>
                          🧩 Skills
                          {skillsEnabled && skillsMode === 'manual' && selectedSkillIds.length > 0
                            ? `（已选 ${selectedSkillIds.length}）`
                            : skillsEnabled && enabledSkills.length > 0
                              ? `（${enabledSkills.length}）`
                              : ''}
                        </span>
                      </label>
                      {skillsEnabled && (
                        <button
                          type="button"
                          className={`skills-mode-btn ${skillsPickerOpen ? 'active' : ''}`}
                          aria-expanded={skillsPickerOpen}
                          aria-label="配置 Skills 模式"
                          onClick={() => setSkillsPickerOpen((prev) => !prev)}>
                          {skillsMode === 'manual' ? '指定 ▾' : '自动 ▾'}
                        </button>
                      )}
                    </div>
                    {ragEnabled && (
                      <div className="knowledge-qa-selects">
                        <label>
                          <span className="sr-only">策略</span>
                          <select
                            aria-label="教材问答召回策略"
                            value={ragCoverage}
                            onChange={(e) => setRagCoverage(e.target.value as CoverageMode)}>
                            <option value="global">全局相关性</option>
                            <option value="per_document">跨教材对比</option>
                          </select>
                        </label>
                        <label>
                          <span className="sr-only">模式</span>
                          <select
                            aria-label="教材问答检索模式"
                            value={ragMode}
                            onChange={(e) => setRagMode(e.target.value as RetrievalMode)}>
                            <option value="hybrid">混合检索</option>
                            <option value="keyword">仅关键词</option>
                          </select>
                        </label>
                      </div>
                    )}
                  </section>

                  <div className="composer-actions">
                    {!modelConfigured && <p className="composer-hint">配置 DeepSeek API Key 后即可发送。</p>}
                    {active.submittingError && (
                      <p role="alert" className="composer-hint composer-error">
                        <span>{active.submittingError}</span>
                        {onOpenMcpPanel && isMcpManageableError(active.submittingErrorCode) && (
                          <button type="button" className="inline-jump-btn" onClick={onOpenMcpPanel}>
                            前往 MCP 服务
                          </button>
                        )}
                      </p>
                    )}
                    {tooLong && <p role="alert" className="composer-hint">问题最多 {MESSAGE_LIMIT.toLocaleString('zh-CN')} 字符。</p>}
                    {trackedRunId && active.activeStatus && !isTerminal(active.activeStatus) && (
                      <button
                        type="button"
                        className="stop-btn"
                        onClick={() => void stop()}
                        disabled={cancellations.current.has(active.session_id) || stopping === active.session_id
                          || active.activeStatus !== 'running'}>
                        停止生成
                      </button>
                    )}
                    <button type="submit" className="send-btn" disabled={!canSend}>
                      发送
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
