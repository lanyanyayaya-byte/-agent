import { invoke } from '@tauri-apps/api/core';
import { useCallback, useEffect, useState, type FormEvent } from 'react';
import {
  ApiRequestError,
  type ApiClient,
  type CoverageMode,
  type DocumentDetails,
  type DocumentSummary,
  type KnowledgeSearchResponse,
  type RetrievalMode,
} from './api/apiClient';
import TextbookIndex from './TextbookIndex';

type ImportRequest = { action: 'select'; document_id?: string } | { action: 'retry'; import_id: string };
type ImportOutcome = 'imported' | 'duplicate' | 'updated' | 'unchanged';
type ImportResult = {
  import_id: string;
  document_id: string;
  document_version_id: string;
  outcome: ImportOutcome;
};

const outcomes: Record<ImportOutcome, string> = {
  imported: '教材已导入。',
  duplicate: '此文件内容已在教材库中，未新增教材。',
  updated: '教材已更新，新增一个版本。',
  unchanged: '文件内容与最新版本相同，未新增版本。',
};
const formats = { pdf: 'PDF', docx: 'DOCX', md: 'Markdown' };
const statuses = { imported: '已导入' };
const dateTime = new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' });
const number = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 1 });

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

function DateTime({ value }: { value: string }) {
  return <time dateTime={value}>{dateTime.format(new Date(value))}</time>;
}

export default function LibraryPanel({
  client, active, siliconflowConfigured = false,
}: { client?: ApiClient; active: boolean; siliconflowConfigured?: boolean }) {
  const [items, setItems] = useState<DocumentSummary[]>();
  const [revision, setRevision] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [selectedId, setSelectedId] = useState<string>();
  const [confirmDelete, setConfirmDelete] = useState<DocumentSummary>();
  const [pending, setPending] = useState('');
  const [notice, setNotice] = useState('');
  const [actionError, setActionError] = useState('');
  const [retryId, setRetryId] = useState<string>();

  useEffect(() => {
    if (!client || !active) return;
    const controller = new AbortController();
    setLoading(true);
    setLoadError('');
    void client.listDocuments(controller.signal).then((documents) => {
      if (controller.signal.aborted) return;
      setItems(documents);
      setSelectedId((current) => documents.some((document) => document.document_id === current) ? current : undefined);
      setLoading(false);
    }).catch((error) => {
      if (controller.signal.aborted) return;
      setLoadError(errorMessage(error, '无法加载教材列表，请检查后端连接或刷新列表。'));
      setLoading(false);
    });
    return () => controller.abort();
  }, [active, client, revision]);

  const refresh = useCallback(() => {
    setRevision((value) => value + 1);
  }, []);

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

  async function deleteDocument(document: DocumentSummary) {
    if (!client || pending) return;
    setNotice('');
    setActionError('');
    setPending('正在删除教材…');
    try {
      await client.deleteDocument(document.document_id, new AbortController().signal);
      setItems((previous) => previous?.filter((item) => item.document_id !== document.document_id));
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
        <button type="button" disabled={!canSelect} onClick={() => void importDocument({ action: 'select' })}>
          导入教材
        </button>
      </div>
      <p className="library-hint">支持 PDF、DOCX 和 Markdown，每次选择一个文件。</p>
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
      {items?.length === 0 && !loading && !loadError && <p>还没有教材。点击「导入教材」添加第一份文件。</p>}
      {!!items?.length && (
        <div className="library-table">
          <table aria-label="教材列表">
            <thead>
              <tr>
                <th scope="col">名称</th><th scope="col">格式</th><th scope="col">最新版本大小</th>
                <th scope="col">导入时间</th><th scope="col">版本数</th><th scope="col">管理状态</th><th scope="col">操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.document_id}>
                  <th scope="row"><bdi>{item.name}</bdi></th>
                  <td>{formats[item.format]}</td>
                  <td>{fileSize(item.latest_size_bytes)}</td>
                  <td><DateTime value={item.created_at} /></td>
                  <td>{item.version_count}</td>
                  <td>{statuses[item.management_status]}</td>
                  <td>
                    <div className="library-actions">
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
              ))}
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
          active={active} blocked={!canChange} siliconflowConfigured={siliconflowConfigured} refreshLibrary={refresh} />
      )}
      {!!items?.length && client && (
        <LibrarySearchPanel client={client} documents={items} />
      )}
    </section>
  );
}

const channelStatusLabels: Record<string, string> = {
  hybrid: '混合检索（关键词 + 语义向量）',
  keyword_only: '仅关键词检索（离线或向量未就绪降级）',
  vector_only: '仅语义向量检索',
  partial: '部分混合检索',
  unavailable: '当前范围内无可用索引',
};

const evidenceStatusLabels: Record<string, string> = {
  evidence: '已召回证据',
  no_evidence: '未找到可靠证据',
  not_returned: '有候选但未入围 Top-K',
  not_assessed: '全局模式未单独评估',
  unavailable: '索引未就绪',
};

function LibrarySearchPanel({ client, documents }: { client: ApiClient; documents: DocumentSummary[] }) {
  const [query, setQuery] = useState('');
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>('hybrid');
  const [coverageMode, setCoverageMode] = useState<CoverageMode>('global');
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [result, setResult] = useState<KnowledgeSearchResponse>();

  function toggleDoc(docId: string) {
    setSelectedDocs((prev) => prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId]);
  }

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || searching) return;
    setSearching(true);
    setSearchError('');
    try {
      const res = await client.searchKnowledge({
        query: trimmed,
        top_k: 8,
        document_ids: selectedDocs.length > 0 ? selectedDocs : null,
        retrieval_mode: retrievalMode,
        coverage_mode: coverageMode,
      }, new AbortController().signal);
      setResult(res);
    } catch (error) {
      setSearchError(error instanceof ApiRequestError ? error.message : '无法完成本地教材检索，请检查索引状态后重试。');
    } finally {
      setSearching(false);
    }
  }

  return (
    <section className="library-search" aria-label="本地教材检索">
      <div className="library-search-header">
        <h3>本地教材检索</h3>
        <span className="library-hint">直接检索本地已发布的关键词与向量索引，无需调用 DeepSeek，毫秒级定位原文出处。</span>
      </div>
      <form className="library-search-form" onSubmit={(e) => void handleSearch(e)}>
        <div className="library-search-bar">
          <input
            type="search"
            aria-label="检索教材内容"
            placeholder="输入术语、定理、公式或自然语言问题（如：最大似然估计、MLE）"
            maxLength={256}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" disabled={searching || !query.trim()}>
            {searching ? '正在检索…' : '检索教材'}
          </button>
        </div>
        <div className="library-search-options">
          <label>
            检索模式
            <select aria-label="检索模式" value={retrievalMode}
              onChange={(e) => setRetrievalMode(e.target.value as RetrievalMode)}>
              <option value="hybrid">混合检索（FTS5 + BGE-M3）</option>
              <option value="keyword">仅关键词检索（离线 FTS5）</option>
              <option value="vector">仅语义向量检索</option>
            </select>
          </label>
          <label>
            召回策略
            <select aria-label="召回策略" value={coverageMode}
              onChange={(e) => setCoverageMode(e.target.value as CoverageMode)}>
              <option value="global">全局相关性（Top-K）</option>
              <option value="per_document">跨教材对比（分教材均衡配额）</option>
            </select>
          </label>
        </div>
        {documents.length > 1 && (
          <div className="library-doc-filters" role="group" aria-label="限定检索教材范围">
            <span className="filter-label">限定教材：</span>
            <button type="button" className={`filter-chip ${selectedDocs.length === 0 ? 'active' : ''}`}
              onClick={() => setSelectedDocs([])}>
              全部（{documents.length}）
            </button>
            {documents.map((doc) => {
              const active = selectedDocs.includes(doc.document_id);
              return (
                <button key={doc.document_id} type="button"
                  className={`filter-chip ${active ? 'active' : ''}`}
                  aria-pressed={active}
                  onClick={() => toggleDoc(doc.document_id)}>
                  <bdi>{doc.name}</bdi>
                </button>
              );
            })}
          </div>
        )}
      </form>
      {searchError && <p role="alert">{searchError}</p>}
      {result && (
        <div className="library-search-results" aria-live="polite">
          <div className="search-summary-bar">
            <span className="badge badge-channel">
              {channelStatusLabels[result.channel_status] ?? result.channel_status}
            </span>
            <span>共命中 <strong>{result.hits.length}</strong> 条已核验教材证据</span>
            <span className="search-timing">
              耗时 {(result.timings.scope_ms + result.timings.fts_ms + result.timings.vector_ms + result.timings.fusion_ms + result.timings.source_ms).toFixed(1)} ms
            </span>
          </div>
          {result.diagnostic && (
            <p className="search-diagnostic" role="status">
              降级提示：{result.diagnostic.message}（<code>{result.diagnostic.code}</code>）
            </p>
          )}
          {result.coverage_mode === 'per_document' && result.documents.length > 0 && (
            <div className="coverage-grid" aria-label="跨教材覆盖统计">
              {result.documents.map((doc) => (
                <div key={doc.document_id} className={`coverage-card status-${doc.evidence_status}`}>
                  <strong><bdi>{doc.textbook_name ?? doc.document_id}</bdi></strong>
                  <span>{evidenceStatusLabels[doc.evidence_status] ?? doc.evidence_status}（{doc.returned_count} 条）</span>
                </div>
              ))}
            </div>
          )}
          {result.hits.length === 0 ? (
            <p className="search-empty">未检索到匹配的教材证据。请确认教材已建立索引，或尝试更换关键词。</p>
          ) : (
            <ol className="evidence-list" aria-label="教材检索结果">
              {result.hits.map((hit, idx) => (
                <li key={hit.evidence_id} className="evidence-card">
                  <div className="evidence-header">
                    <span className="evidence-index">证据 #{idx + 1}</span>
                    <strong className="evidence-book"><bdi>{hit.textbook_name}</bdi></strong>
                    <span className="badge badge-format">{formats[hit.format]}</span>
                    <span className="badge badge-location">{hit.location_summary}</span>
                    <span className="evidence-score">RRF {hit.fusion_score.toFixed(4)}</span>
                  </div>
                  {hit.section_path.length > 0 && (
                    <div className="evidence-section">
                      章节：{hit.section_path.join(' / ')}
                    </div>
                  )}
                  {hit.warnings.length > 0 && (
                    <div className="evidence-warnings" role="note">
                      ⚠️ 公式与字形核验提示：{hit.warnings.map((w) => `${w.code}（${w.message}）`).join('；')}
                      ——引用数学表达式时请核对原教材对应页。
                    </div>
                  )}
                  <pre className="evidence-snippet">{hit.body_text}</pre>
                  <div className="evidence-footer">
                    {hit.channel_ranks.map((r) => (
                      <span key={r.channel} className="rank-pill">
                        {r.channel === 'keyword' ? '关键词' : '语义向量'} #{r.rank}（{r.score.toFixed(3)}）
                      </span>
                    ))}
                    <span className="evidence-id-tag" title={hit.evidence_id}>
                      已核验原件 · {hit.evidence_id.slice(0, 22)}…
                    </span>
                  </div>
                </li>
              ))}
            </ol>
          )}
        </div>
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
                  {details.versions.map((version, index) => (
                    <tr key={version.document_version_id}>
                      <th scope="row">第 {index + 1} 版</th>
                      <td className="library-hash">{version.document_version_id}<br />{version.content_sha256}</td>
                      <td>{fileSize(version.size_bytes)}</td>
                      <td><DateTime value={version.created_at} /></td>
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
