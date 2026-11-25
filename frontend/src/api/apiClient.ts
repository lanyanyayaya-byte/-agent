import { invoke } from '@tauri-apps/api/core';
import { parseRunEvent, type ParseWarning, type RunEvent, type StoredCitation } from './chatEvents';
import { readSse } from './sse';

export type { ParseWarning, StoredCitation };
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

export type ToolCallSummary = {
  tool_call_id: string;
  tool_name: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  summary: string | null;
  error_code: string | null;
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
  tool_calls?: ToolCallSummary[];
  citations?: StoredCitation[];
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
    ): Promise<void> {
      const response = await request(`/sessions/${encodeURIComponent(sessionId)}/runs/stream`, 'POST', signal, {
        accept: 'text/event-stream',
        body: {
          client_request_id: clientRequestId,
          message,
          ...(knowledge ? { knowledge } : {}),
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
