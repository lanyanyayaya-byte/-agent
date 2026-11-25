import { invoke } from '@tauri-apps/api/core';
import { useCallback, useEffect, useState } from 'react';
import {
  ApiRequestError,
  type ApiClient,
  type BatchImportItem,
  type DocumentDetails,
  type DocumentIndexStatus,
  type IndexJobStage,
  type KnowledgeIndexOverview,
  type KnowledgeSettings,
} from './api/apiClient';
import ChunkBrowser from './ChunkBrowser';
import RetrievalLab from './RetrievalLab';
import TextbookIndex from './TextbookIndex';

type LibraryTab = 'manage' | 'lab' | 'chunks';

type ImportRequest = { action: 'select'; document_id?: string } | { action: 'retry'; import_id: string };
type ImportOutcome = 'imported' | 'duplicate' | 'updated' | 'unchanged';
type ImportResult = {
  import_id: string;
  document_id: string;
  document_version_id: string;
  outcome: ImportOutcome;
};

const outcomes: Record<ImportOutcome, string> = {
  imported: '教材已导入，已加入后台自动索引队列。',
  duplicate: '此文件内容已在教材库中，未新增教材。',
  updated: '教材已更新，新增一个版本并已加入自动索引队列。',
  unchanged: '文件内容与最新版本相同，未新增版本。',
};
const formats = { pdf: 'PDF', docx: 'DOCX', md: 'Markdown' };
const statuses = { imported: '已导入' };
const stageLabels: Record<IndexJobStage, string> = {
  parsing: '解析中',
  chunking: '分块中',
  keyword_indexing: 'BM25索引',
  embedding: '向量化中',
  publishing: '发布索引',
};
const dateTime = new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' });
const number = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 });
const ACTIVE_OVERVIEW_POLL_MS = 2500;

function fileSize(bytes: number | null) {
  if (bytes === null) return '—';
  if (bytes >= 1024 ** 2) return `${number.format(bytes / 1024 ** 2)} MiB`;
  if (bytes >= 1024) return `${number.format(bytes / 1024)} KiB`;
  return `${bytes} B`;
}

function errorCode(error: unknown) {
  if (error instanceof ApiRequestError) return error.code;
  if (typeof error === 'string') return error;
  if (error && typeof error === 'object' && 'code' in error && typeof error.code === 'string') return error.code;
}

function errorMessage(error: unknown, fallback: string) {
  switch (errorCode(error)) {
    case 'KNOWLEDGE_STORE_UNAVAILABLE': return '教材知识库暂不可用，请稍后重试。';
    case 'SIDECAR_NOT_READY':
    case 'BACKEND_SIDECAR_UNAVAILABLE':
    case 'BACKEND_SHUTTING_DOWN': return '桌面后端暂不可用，请检查后端连接后重试。';
    case 'TEXTBOOK_TOO_LARGE': return '文件超过 100 MiB，请选择更小的教材文件。';
    case 'INVALID_TEXTBOOK_FORMAT': return '文件格式不受支持或内容与扩展名不符。请选择有效的 PDF、DOCX 或 UTF-8 Markdown 文件。';
    case 'TEXTBOOK_FORMAT_MISMATCH': return '更新文件的格式必须与原教材一致。';
    case 'TEXTBOOK_NOT_FOUND':
    case 'TEXTBOOK_REMOVED': return '该教材已不存在，请刷新列表。';
    case 'IMPORT_PICKER_BUSY': return '已有文件选择窗口打开，请先完成当前选择。';
    case 'IMPORT_PICKER_FAILED': return '无法打开文件选择窗口，请重试。';
    case 'IMPORT_UNSUPPORTED_PLATFORM': return '教材文件选择目前仅支持 macOS。';
    case 'IMPORT_BUSY': return '正在处理另一项教材导入，请稍后重试。';
    case 'IMPORT_DISK_FULL': return '磁盘空间不足，释放空间后可重试导入。';
    case 'INVALID_IMPORT_SOURCE': return '无法读取所选文件，请选择可访问的本地文件。';
    case 'IMPORT_SOURCE_CHANGED': return '导入期间文件发生变化，请重试。';
    case 'IMPORT_RETRY_NOT_FOUND': return '上次导入的暂存记录已失效，请刷新列表并重新选择文件。';
    case 'IMPORT_REQUEST_TIMEOUT':
    case 'IMPORT_RESPONSE_UNCERTAIN':
    case 'IMPORT_RETRY_REQUIRED':
    case 'IMPORT_HOST_AUTH_FAILED':
    case 'IMPORT_SOURCE_NOT_FOUND': return '暂未确认上次导入的结果，请重试本次导入。';
    default: return fallback;
  }
}

function isJobActive(doc: DocumentIndexStatus) {
  const st = doc.latest_job?.status;
  return st === 'queued' || st === 'running' || st === 'retry_wait';
}

function getRowOverallBadge(doc: DocumentIndexStatus): { label: string; tone: 'ready' | 'keyword' | 'running' | 'failed' | 'pending' | 'muted' } {
  if (isJobActive(doc)) {
    const stage = doc.latest_job ? stageLabels[doc.latest_job.stage] : '处理中';
    if (doc.latest_job?.status === 'retry_wait') {
      return { label: `等待重试 · ${stage}`, tone: 'running' };
    }
    return { label: `索引中 · ${stage}`, tone: 'running' };
  }
  if (doc.latest_job?.status === 'failed' && doc.published_readiness !== 'ready_hybrid') {
    return { label: '索引失败', tone: 'failed' };
  }
  if (doc.published_readiness === 'ready_hybrid') {
    return { label: '混合就绪', tone: 'ready' };
  }
  if (doc.published_readiness === 'ready_keyword') {
    return { label: '仅关键词', tone: 'keyword' };
  }
  if (doc.published_readiness === 'unsearchable') {
    return { label: '无正文', tone: 'muted' };
  }
  return { label: '未索引', tone: 'pending' };
}

function getParseLabel(doc: DocumentIndexStatus) {
  if (doc.parse_status === 'ready') {
    if (doc.page_count !== null && doc.page_count > 0) {
      return `已解析 · ${doc.page_count} 页`;
    }
    return '已解析';
  }
  if (doc.parse_status === 'writing') return '解析中';
  if (doc.parse_status === 'failed') return '解析失败';
  return '未解析';
}

function getJobDiagnosticText(doc: DocumentIndexStatus) {
  const job = doc.latest_job;
  if (job) {
    if (job.status === 'running' || job.status === 'queued' || job.status === 'retry_wait') {
      const stage = stageLabels[job.stage] ?? job.stage;
      if (job.total_work !== null && job.total_work > 0) {
        return `${stage} (${job.completed_work}/${job.total_work})`;
      }
      return job.status === 'queued' ? `排队中 · ${stage}` : `${stage}…`;
    }
    if (job.status === 'failed') {
      return job.error_message || job.error_code || '任务失败';
    }
    if (job.status === 'cancelled') {
      return '已取消';
    }
  }
  if (doc.embedding_status === 'waiting_key') {
    return '等待配置 SiliconFlow Key';
  }
  if (doc.embedding_status === 'disabled') {
    return '未开启云端向量化';
  }
  if (doc.embedding_status === 'partial') {
    return `部分向量化 (${doc.embedded_count}/${doc.chunk_count ?? 0})`;
  }
  if (doc.published_readiness === 'ready_hybrid') {
    return doc.embedding_model ? `BGE-M3 · ${doc.embedding_dimension ?? 1024}d` : '混合检索就绪';
  }
  if (doc.published_readiness === 'unsearchable') {
    return '无可用文本层';
  }
  return '待索引';
}

function DateTime({ value }: { value: string }) {
  return <time dateTime={value}>{dateTime.format(new Date(value))}</time>;
}

export default function LibraryPanel({
  client, active, siliconflowConfigured = false,
}: { client?: ApiClient; active: boolean; siliconflowConfigured?: boolean }) {
  const [overview, setOverview] = useState<KnowledgeIndexOverview>();
  const [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [selectedId, setSelectedId] = useState<string>();
  const [confirmDelete, setConfirmDelete] = useState<DocumentIndexStatus>();
  const [pending, setPending] = useState('');
  const [rowBusyId, setRowBusyId] = useState<string>();
  const [notice, setNotice] = useState('');
  const [actionError, setActionError] = useState('');
  const [retryId, setRetryId] = useState<string>();
  const [confirmCloudEnable, setConfirmCloudEnable] = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);
  const [activeTab, setActiveTab] = useState<LibraryTab>('manage');
  const [chunkTarget, setChunkTarget] = useState<{ documentId?: string; page?: number }>({});

  const loadOverview = useCallback(async (source: ApiClient, signal: AbortSignal, silent: boolean) => {
    if (!silent) {
      setLoading(true);
      setLoadError('');
    }
    try {
      const next = await source.getKnowledgeIndexStatus(signal);
      if (signal.aborted) return;
      setOverview(next);
      setSelectedId((current) => next.documents.some((doc) => doc.document_id === current) ? current : undefined);
      setLoadError('');
    } catch (error) {
      if (signal.aborted) return;
      if (!silent) {
        setLoadError(errorMessage(error, '无法加载教材列表，请检查后端连接或刷新列表。'));
      }
    } finally {
      if (!signal.aborted && !silent) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    if (!client || !active) return;
    const controller = new AbortController();
    void loadOverview(client, controller.signal, false);
    return () => controller.abort();
  }, [active, client, loadOverview, revision]);

  const hasActiveJobs = !!overview && (overview.in_progress > 0 || overview.active_job_id !== null);

  useEffect(() => {
    if (!client || !active || !hasActiveJobs) return;
    const controller = new AbortController();
    const timer = setInterval(() => {
      void loadOverview(client, controller.signal, true);
    }, ACTIVE_OVERVIEW_POLL_MS);
    return () => {
      clearInterval(timer);
      controller.abort();
    };
  }, [active, client, hasActiveJobs, loadOverview]);

  const refresh = useCallback(() => {
    setRevision((value) => value + 1);
  }, []);

  async function updateSettings(patch: Partial<KnowledgeSettings>, successText?: string) {
    if (!client || savingSettings) return;
    setSavingSettings(true);
    setActionError('');
    try {
      const updated = await client.updateKnowledgeSettings(patch, new AbortController().signal);
      setOverview((prev) => prev ? { ...prev, settings: updated } : prev);
      setConfirmCloudEnable(false);
      if (successText) setNotice(successText);
      refresh();
    } catch (error) {
      setActionError(errorMessage(error, '无法保存知识库索引设置，请稍后重试。'));
    } finally {
      setSavingSettings(false);
    }
  }

  async function importDocument(request: ImportRequest) {
    if (!client || pending) return;
    setNotice('');
    setActionError('');
    setPending(request.action === 'select' && request.document_id ? '正在更新教材…' : '正在导入教材…');
    try {
      const result = await invoke<ImportResult | null>('select_and_import_document', { request });
      if (result === null) return;
      setRetryId(undefined);
      setNotice(outcomes[result.outcome]);
      setSelectedId(result.document_id);
      refresh();
    } catch (error) {
      const importId = error && typeof error === 'object' && 'import_id' in error ? error.import_id : undefined;
      setRetryId(typeof importId === 'string' ? importId : undefined);
      if (errorCode(error) !== 'IMPORT_CANCELLED') {
        setActionError(errorMessage(error, '无法完成教材导入，请检查文件和后端连接后重试。'));
      }
    } finally {
      setPending('');
    }
  }

  async function importMultipleDocuments() {
    if (!client || pending) return;
    setNotice('');
    setActionError('');
    setPending('正在批量导入教材…');
    try {
      const items = await invoke<BatchImportItem[] | null>('select_and_import_documents');
      if (!items || items.length === 0) return;
      const imported = items.filter((item) => item.ok && item.result?.outcome === 'imported').length;
      const duplicates = items.filter((item) => item.ok && item.result?.outcome === 'duplicate').length;
      const failed = items.filter((item) => !item.ok);
      const parts = [`批量导入完成：新增 ${imported} 本`];
      if (duplicates > 0) parts.push(`重复跳过 ${duplicates} 本`);
      if (failed.length > 0) parts.push(`失败 ${failed.length} 本`);
      setNotice(`${parts.join('，')}。`);
      if (failed.length > 0) {
        const failedNames = failed.slice(0, 3).map((f) => f.filename).join('、');
        setActionError(`部分文件导入失败：${failedNames}${failed.length > 3 ? ' 等' : ''}。`);
      }
      refresh();
    } catch (error) {
      if (errorCode(error) !== 'IMPORT_CANCELLED') {
        setActionError(errorMessage(error, '无法完成批量导入，请检查文件和后端连接后重试。'));
      }
    } finally {
      setPending('');
    }
  }

  async function handleIndexAll() {
    if (!client || pending) return;
    setNotice('');
    setActionError('');
    setPending('正在提交补全索引队列…');
    try {
      const result = await client.indexAllDocuments(new AbortController().signal);
      if (result.queued_count === 0) {
        setNotice(`所有教材索引均已完整或在队列中（跳过已完整 ${result.skipped_complete} 本）。`);
      } else {
        const modeText = result.mode === 'hybrid' ? '混合向量索引' : '本地关键词索引';
        setNotice(`已加入${modeText}队列 ${result.queued_count} 本（跳过已就绪 ${result.skipped_complete} 本）。`);
      }
      refresh();
    } catch (error) {
      setActionError(errorMessage(error, '无法提交一键补全索引，请检查后端状态后重试。'));
    } finally {
      setPending('');
    }
  }

  async function handleRebuildStale() {
    if (!client || pending) return;
    setNotice('');
    setActionError('');
    setPending('正在提交待升级教材重建队列…');
    try {
      const result = await client.rebuildStaleIndexes(new AbortController().signal);
      if (result.queued_count === 0) {
        setNotice('当前没有需要升级解析/切片版本的教材。');
      } else {
        setNotice(`已加入升级重建队列 ${result.queued_count} 本（未变化切片的向量将自动复用缓存）。`);
      }
      refresh();
    } catch (error) {
      setActionError(errorMessage(error, '无法提交待升级教材重建，请检查后端状态后重试。'));
    } finally {
      setPending('');
    }
  }

  function handleInspectChunk(documentId: string, page?: number) {
    setChunkTarget({ documentId, page });
    setActiveTab('chunks');
  }

  async function handleRowIndexAction(doc: DocumentIndexStatus, action: 'rebuild' | 'cancel') {
    if (!client || rowBusyId) return;
    setNotice('');
    setActionError('');
    setRowBusyId(doc.document_id);
    try {
      if (action === 'cancel') {
        await client.cancelDocumentIndexJob(doc.document_id, new AbortController().signal);
        setNotice(`已请求取消《${doc.name}》的索引任务。`);
      } else {
        const forceRebuild = doc.published_readiness === 'ready_hybrid';
        await client.rebuildDocumentIndex(doc.document_id, forceRebuild, new AbortController().signal);
        setNotice(forceRebuild ? `已将《${doc.name}》加入重建索引队列。` : `已将《${doc.name}》加入补全索引队列。`);
      }
      refresh();
    } catch (error) {
      setActionError(errorMessage(error, `无法更新《${doc.name}》的索引任务，请稍后重试。`));
    } finally {
      setRowBusyId(undefined);
    }
  }

  async function deleteDocument(document: DocumentIndexStatus) {
    if (!client || pending) return;
    setNotice('');
    setActionError('');
    setPending('正在删除教材…');
    try {
      await client.deleteDocument(document.document_id, new AbortController().signal);
      setOverview((previous) => previous
        ? {
          ...previous,
          total_documents: Math.max(0, previous.total_documents - 1),
          documents: previous.documents.filter((item) => item.document_id !== document.document_id),
        }
        : previous);
      setSelectedId((previous) => previous === document.document_id ? undefined : previous);
      setConfirmDelete(undefined);
      setNotice('教材已删除。');
      refresh();
    } catch (error) {
      setActionError(errorMessage(error, '无法确认删除结果，请刷新列表或重试删除。'));
    } finally {
      setPending('');
    }
  }

  const items = overview?.documents;
  const settings = overview?.settings;
  const effectiveSfConfigured = overview?.siliconflow_configured ?? siliconflowConfigured;
  const missingCount = overview?.missing_for_completion ?? 0;
  const staleCount = overview?.stale_count ?? 0;
  const busy = !!pending;
  const canChange = !!client && !busy && !confirmDelete;
  const canSelect = canChange && !retryId;

  return (
    <section className="library" aria-labelledby="library-heading">
      <div className="library-toolbar">
        <h2 id="library-heading">教材知识库</h2>
        <button type="button" className="secondary-button" disabled={!client || loading} onClick={refresh}>
          刷新列表
        </button>
        {staleCount > 0 && (
          <button
            type="button"
            className="secondary-button"
            disabled={!canChange || !overview}
            onClick={() => void handleRebuildStale()}
          >
            重建待升级索引（{staleCount}）
          </button>
        )}
        <button
          type="button"
          className="secondary-button"
          disabled={!canChange || !overview || missingCount === 0}
          onClick={() => void handleIndexAll()}
        >
          {missingCount > 0 ? `一键补全索引（${missingCount}）` : '全部已索引'}
        </button>
        <button type="button" className="secondary-button" disabled={!canSelect} onClick={() => void importMultipleDocuments()}>
          批量导入教材
        </button>
        <button type="button" disabled={!canSelect} onClick={() => void importDocument({ action: 'select' })}>
          导入教材
        </button>
      </div>

      <div className="library-subtabs" role="tablist" aria-label="教材知识库功能切换">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'manage'}
          className={`library-subtab ${activeTab === 'manage' ? 'active' : ''}`}
          onClick={() => setActiveTab('manage')}
        >
          📚 教材管理与索引{overview ? `（${overview.total_documents}）` : ''}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'lab'}
          className={`library-subtab ${activeTab === 'lab' ? 'active' : ''}`}
          disabled={!client || !items?.length}
          onClick={() => setActiveTab('lab')}
        >
          🔬 RAG 检索实验室
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'chunks'}
          className={`library-subtab ${activeTab === 'chunks' ? 'active' : ''}`}
          disabled={!client || !items?.length}
          onClick={() => setActiveTab('chunks')}
        >
          🧩 Chunk 原文浏览器
        </button>
      </div>

      {activeTab === 'lab' && client && !!items?.length && (
        <RetrievalLab
          client={client}
          documents={items}
          defaultRerank={!!settings?.cloud_rerank_enabled && !!settings?.cloud_embedding_enabled}
          onInspectChunk={handleInspectChunk}
        />
      )}

      {activeTab === 'chunks' && client && !!items?.length && (
        <ChunkBrowser
          client={client}
          documents={items}
          initialDocumentId={chunkTarget.documentId}
          initialPage={chunkTarget.page}
        />
      )}

      {activeTab === 'manage' && (
        <>
          <p className="library-hint">支持 PDF、DOCX 和 Markdown，导入后自动建立解析、BM25 关键词与 BGE-M3 向量索引。</p>

      {overview && (
        <div className="library-summary-grid" aria-label="知识库索引统计">
          <div className="summary-kpi">
            <span className="summary-kpi-label">总教材</span>
            <strong className="summary-kpi-value">{overview.total_documents}</strong>
          </div>
          <div className="summary-kpi tone-ready">
            <span className="summary-kpi-label">混合就绪</span>
            <strong className="summary-kpi-value">{overview.ready_hybrid}</strong>
          </div>
          <div className="summary-kpi tone-keyword">
            <span className="summary-kpi-label">仅关键词</span>
            <strong className="summary-kpi-value">{overview.ready_keyword}</strong>
          </div>
          <div className="summary-kpi tone-running">
            <span className="summary-kpi-label">索引中</span>
            <strong className="summary-kpi-value">{overview.in_progress}</strong>
          </div>
          <div className="summary-kpi tone-pending">
            <span className="summary-kpi-label">未索引</span>
            <strong className="summary-kpi-value">{overview.pending}</strong>
          </div>
          <div className="summary-kpi tone-failed">
            <span className="summary-kpi-label">失败</span>
            <strong className="summary-kpi-value">{overview.failed}</strong>
          </div>
        </div>
      )}

      {settings && (
        <div className="library-settings-bar" aria-label="全局索引设置">
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.auto_index_on_import}
              disabled={!client || savingSettings}
              onChange={(e) => void updateSettings({ auto_index_on_import: e.target.checked })}
            />
            <span>导入后自动建立索引</span>
          </label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.cloud_embedding_enabled}
              disabled={!client || savingSettings}
              onChange={(e) => {
                if (e.target.checked && !settings.cloud_embedding_enabled) {
                  setConfirmCloudEnable(true);
                } else {
                  setConfirmCloudEnable(false);
                  void updateSettings({ cloud_embedding_enabled: false }, '已关闭云端自动向量化，后续仅建立本地关键词索引。');
                }
              }}
            />
            <span>允许云端向量化（BGE-M3）</span>
          </label>
          <label className="settings-toggle">
            <input
              type="checkbox"
              checked={settings.cloud_rerank_enabled}
              disabled={!client || savingSettings || !settings.cloud_embedding_enabled}
              onChange={(e) => void updateSettings({ cloud_rerank_enabled: e.target.checked })}
            />
            <span>启用语义精排（BGE-Reranker-v2-M3）</span>
          </label>
          {!effectiveSfConfigured && (
            <span className="settings-sf-hint">
              未配置 SiliconFlow Key：当前仅建立本地 BM25 索引，配置 Key 后可一键补全向量。
            </span>
          )}
        </div>
      )}

      {confirmCloudEnable && (
        <div className="library-cloud-consent" role="group" aria-label="确认开启全局云端向量化">
          <p>
            开启后，新导入教材及「一键补全索引」将自动把教材分块（Chunk）文本发送至 SiliconFlow API 生成 BGE-M3 向量（原始文件不会整本上传），后续无需再逐本确认。
          </p>
          <div className="library-actions">
            <button
              type="button"
              autoFocus
              disabled={savingSettings}
              onClick={() => void updateSettings({ cloud_embedding_enabled: true }, '已开启全局云端向量化授权。')}
            >
              确认开启云端向量化
            </button>
            <button type="button" className="secondary-button" disabled={savingSettings} onClick={() => setConfirmCloudEnable(false)}>
              暂不开启
            </button>
          </div>
        </div>
      )}

      {!client && <p role="status">后端连接就绪后可管理教材。</p>}
      {loading && client && <p role="status">正在加载教材列表…</p>}
      {loadError && client && loadError !== actionError && <p role="alert">{loadError}</p>}
      {pending && <p role="status">{pending}</p>}
      {notice && <p role="status">{notice}</p>}
      {actionError && <p role="alert">{actionError}</p>}
      {retryId && (
        <button type="button" className="secondary-button" disabled={!canChange}
          onClick={() => void importDocument({ action: 'retry', import_id: retryId })}>
          重试本次导入
        </button>
      )}
      {items?.length === 0 && !loading && !loadError && <p>还没有教材。点击「导入教材」或「批量导入教材」添加第一份文件。</p>}
      {!!items?.length && (
        <div className="library-table">
          <table aria-label="教材列表">
            <thead>
              <tr>
                <th scope="col">名称</th>
                <th scope="col">解析状态</th>
                <th scope="col">向量覆盖</th>
                <th scope="col">检索状态</th>
                <th scope="col">当前阶段 / 诊断</th>
                <th scope="col">操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => {
                const badge = getRowOverallBadge(item);
                const activeJob = isJobActive(item);
                const rowBusy = rowBusyId === item.document_id;
                const pct = item.vector_coverage_pct !== null ? Math.min(100, Math.max(0, Math.round(item.vector_coverage_pct))) : null;
                return (
                  <tr key={item.document_id}>
                    <th scope="row">
                      <div className="book-title-cell">
                        <bdi>{item.name}</bdi>
                        <span className="book-subline">
                          {formats[item.format]} · {fileSize(item.latest_size_bytes)} · v{item.version_count}
                          {item.needs_rebuild && (
                            <span className="stale-upgrade-pill" title={item.stale_reason ?? '解析或切片版本已升级'}>
                              待升级
                            </span>
                          )}
                        </span>
                      </div>
                    </th>
                    <td>
                      <span className={`parse-pill parse-${item.parse_status}`}>
                        {getParseLabel(item)}
                      </span>
                    </td>
                    <td>
                      {item.chunk_count !== null && item.chunk_count > 0 ? (
                        <div className="coverage-cell">
                          <span className="coverage-numbers">
                            {item.embedded_count} / {item.chunk_count}
                            {pct !== null ? ` (${pct}%)` : ''}
                          </span>
                          <div className="coverage-bar" aria-hidden="true">
                            <div
                              className={`coverage-bar-fill ${pct === 100 ? 'is-full' : pct && pct > 0 ? 'is-partial' : ''}`}
                              style={{ width: `${pct ?? 0}%` }}
                            />
                          </div>
                        </div>
                      ) : (
                        <span className="coverage-empty">—</span>
                      )}
                    </td>
                    <td>
                      <div className="status-pill-group">
                        <span className={`index-status-pill tone-${badge.tone}`}>{badge.label}</span>
                        <span className="channel-mini-flags">
                          <span className={`mini-flag ${item.keyword_status === 'ready' ? 'is-on' : ''}`}>
                            BM25 {item.keyword_status === 'ready' ? '✓' : '·'}
                          </span>
                          <span className={`mini-flag ${item.vector_searchable ? 'is-on' : ''}`}>
                            向量 {item.vector_searchable ? '✓' : '·'}
                          </span>
                        </span>
                      </div>
                    </td>
                    <td className="diagnostic-cell" title={getJobDiagnosticText(item)}>
                      <span className={item.latest_job?.status === 'failed' ? 'diag-error' : 'diag-muted'}>
                        {getJobDiagnosticText(item)}
                      </span>
                    </td>
                    <td>
                      <div className="library-actions">
                        {activeJob ? (
                          <button type="button" disabled={!canChange || rowBusy}
                            aria-label={`取消 ${item.name} 索引`}
                            onClick={() => void handleRowIndexAction(item, 'cancel')}>
                            取消
                          </button>
                        ) : (
                          <button type="button" disabled={!canChange || rowBusy}
                            aria-label={`${item.published_readiness === 'ready_hybrid' ? '重建' : '补全'} ${item.name} 索引`}
                            onClick={() => void handleRowIndexAction(item, 'rebuild')}>
                            {item.latest_job?.status === 'failed' ? '重试' : item.published_readiness === 'ready_hybrid' ? '重建' : '补全'}
                          </button>
                        )}
                        {item.chunk_count !== null && item.chunk_count > 0 && (
                          <button
                            type="button"
                            aria-label={`浏览 ${item.name} 切片`}
                            disabled={!client}
                            onClick={() => handleInspectChunk(item.document_id)}
                          >
                            浏览切片
                          </button>
                        )}
                        <button type="button" aria-label={`查看 ${item.name} 详情`} disabled={!client}
                          aria-expanded={selectedId === item.document_id}
                          onClick={() => setSelectedId(selectedId === item.document_id ? undefined : item.document_id)}>
                          详情与索引
                        </button>
                        <button type="button" aria-label={`更新 ${item.name}`} disabled={!canSelect}
                          onClick={() => void importDocument({ action: 'select', document_id: item.document_id })}>
                          更新
                        </button>
                        <button type="button" className="library-danger" aria-label={`删除 ${item.name}`}
                          disabled={!canChange} onClick={() => { setActionError(''); setConfirmDelete(item); }}>
                          删除
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {confirmDelete && (
        <section className="library-confirm" aria-label="删除确认">
          <h3>确认删除教材</h3>
          <p>将删除《<bdi>{confirmDelete.name}</bdi>》及其全部版本。此操作无法撤销。</p>
          <div className="library-actions">
            <button type="button" autoFocus disabled={busy} onClick={() => setConfirmDelete(undefined)}>取消删除</button>
            <button type="button" className="library-danger" disabled={!client || busy}
              onClick={() => void deleteDocument(confirmDelete)}>
              确认删除
            </button>
          </div>
        </section>
      )}
      {selectedId && (
        <LibraryDetails key={selectedId} documentId={selectedId} client={client} revision={revision}
          active={active} blocked={!canChange} siliconflowConfigured={effectiveSfConfigured} refreshLibrary={refresh} />
      )}
        </>
      )}
    </section>
  );
}

function LibraryDetails({
  client, documentId, revision, active, blocked, siliconflowConfigured, refreshLibrary,
}: {
  client?: ApiClient; documentId: string; revision: number; active: boolean; blocked: boolean;
  siliconflowConfigured: boolean; refreshLibrary: () => void;
}) {
  const [snapshot, setSnapshot] = useState<{ document: DocumentDetails; source: ApiClient; revision: number }>();
  const [error, setError] = useState('');

  useEffect(() => {
    if (!client || !active) return;
    const controller = new AbortController();
    setError('');
    void client.getDocument(documentId, controller.signal).then((document) => {
      if (!controller.signal.aborted) setSnapshot({ document, source: client, revision });
    }).catch((error) => {
      if (controller.signal.aborted) return;
      setSnapshot(undefined);
      setError(errorMessage(error, '无法加载教材详情，请刷新列表后重试。'));
    });
    return () => controller.abort();
  }, [active, client, documentId, revision]);

  const details = snapshot?.document;
  const version = details?.versions.at(-1);
  const currentDetails = snapshot?.source === client && snapshot?.revision === revision;

  return (
    <section className="library-details" aria-labelledby="library-details-heading">
      <h3 id="library-details-heading">{details ? <bdi>{details.name}</bdi> : '教材详情'}</h3>
      {error && <p role="alert">{error}</p>}
      {!details && !error && client && <p role="status">正在加载教材详情…</p>}
      {details && (
        <>
          <dl>
            <div><dt>教材编号</dt><dd>{details.document_id}</dd></div>
            <div><dt>格式</dt><dd>{formats[details.format]}</dd></div>
            <div><dt>管理状态</dt><dd>{statuses[details.management_status]}</dd></div>
            <div><dt>导入时间</dt><dd><DateTime value={details.created_at} /></dd></div>
            <div><dt>更新时间</dt><dd><DateTime value={details.updated_at} /></dd></div>
          </dl>
          {version && currentDetails && (
            <TextbookIndex key={version.document_version_id} client={client} active={active}
              documentId={documentId} versionId={version.document_version_id} versionNumber={details.versions.length}
              blocked={blocked} siliconflowConfigured={siliconflowConfigured} refreshLibrary={refreshLibrary} />
          )}
          {version && !currentDetails && <p role="status">{client ? '正在核对最新教材版本…' : '后端恢复连接后可读取索引状态。'}</p>}
          <h4>版本历史（{details.version_count}）</h4>
          {details.versions.length === 0 ? <p>暂无版本记录。</p> : (
            <div className="library-table">
              <table aria-label="版本历史">
                <thead>
                  <tr>
                    <th scope="col">版本</th><th scope="col">版本编号与 SHA-256</th>
                    <th scope="col">大小</th><th scope="col">创建时间</th>
                  </tr>
                </thead>
                <tbody>
                  {details.versions.map((ver, index) => (
                    <tr key={ver.document_version_id}>
                      <th scope="row">第 {index + 1} 版</th>
                      <td className="library-hash">{ver.document_version_id}<br />{ver.content_sha256}</td>
                      <td>{fileSize(ver.size_bytes)}</td>
                      <td><DateTime value={ver.created_at} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}
