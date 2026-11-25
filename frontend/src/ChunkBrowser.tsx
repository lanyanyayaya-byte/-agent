import { useCallback, useEffect, useState, type FormEvent } from 'react';
import {
  ApiRequestError,
  openTextbookOriginal,
  type ApiClient,
  type ChunkBrowserResponse,
  type ChunkNeighborsResponse,
  type DocumentIndexStatus,
} from './api/apiClient';

const PAGE_SIZE = 20;

export default function ChunkBrowser({
  client,
  documents,
  initialDocumentId,
  initialPage,
}: {
  client: ApiClient;
  documents: DocumentIndexStatus[];
  initialDocumentId?: string;
  initialPage?: number;
}) {
  const [docId, setDocId] = useState<string>(initialDocumentId || documents[0]?.document_id || '');
  const [pageFilter, setPageFilter] = useState<number | undefined>(initialPage);
  const [sectionFilter, setSectionFilter] = useState<string>('');
  const [keywordInput, setKeywordInput] = useState<string>('');
  const [appliedKeyword, setAppliedKeyword] = useState<string>('');
  const [offset, setOffset] = useState<number>(0);

  const [data, setData] = useState<ChunkBrowserResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [openNotice, setOpenNotice] = useState('');

  const [neighborsByIndex, setNeighborsByIndex] = useState<Record<number, ChunkNeighborsResponse>>({});
  const [neighborLoadingIdx, setNeighborLoadingIdx] = useState<number>();
  const [expandedNeighborIdx, setExpandedNeighborIdx] = useState<number>();

  useEffect(() => {
    if (initialDocumentId && initialDocumentId !== docId) {
      setDocId(initialDocumentId);
      setPageFilter(initialPage);
      setSectionFilter('');
      setKeywordInput('');
      setAppliedKeyword('');
      setOffset(0);
      setNeighborsByIndex({});
      setExpandedNeighborIdx(undefined);
    } else if (initialPage !== undefined && initialPage !== pageFilter) {
      setPageFilter(initialPage);
      setOffset(0);
    }
  }, [initialDocumentId, initialPage]);

  useEffect(() => {
    if (!docId && documents.length > 0) {
      setDocId(documents[0].document_id);
    }
  }, [docId, documents]);

  const loadChunks = useCallback(async (signal: AbortSignal) => {
    if (!docId) return;
    setLoading(true);
    setError('');
    try {
      const res = await client.browseDocumentChunks(
        docId,
        {
          page: pageFilter,
          section: sectionFilter || undefined,
          q: appliedKeyword || undefined,
          offset,
          limit: PAGE_SIZE,
        },
        signal,
      );
      if (!signal.aborted) {
        setData(res);
      }
    } catch (err) {
      if (!signal.aborted) {
        setError(err instanceof ApiRequestError ? err.message : '无法加载该教材的切片原文，请确认教材已解析或索引。');
      }
    } finally {
      if (!signal.aborted) {
        setLoading(false);
      }
    }
  }, [client, docId, pageFilter, sectionFilter, appliedKeyword, offset]);

  useEffect(() => {
    const controller = new AbortController();
    void loadChunks(controller.signal);
    return () => controller.abort();
  }, [loadChunks]);

  function handleDocChange(nextId: string) {
    setDocId(nextId);
    setPageFilter(undefined);
    setSectionFilter('');
    setKeywordInput('');
    setAppliedKeyword('');
    setOffset(0);
    setNeighborsByIndex({});
    setExpandedNeighborIdx(undefined);
    setOpenNotice('');
  }

  function handleKeywordSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setAppliedKeyword(keywordInput.trim());
    setOffset(0);
  }

  function handleResetFilters() {
    setPageFilter(undefined);
    setSectionFilter('');
    setKeywordInput('');
    setAppliedKeyword('');
    setOffset(0);
  }

  async function handleToggleNeighbors(chunkIndex: number) {
    if (expandedNeighborIdx === chunkIndex) {
      setExpandedNeighborIdx(undefined);
      return;
    }
    setExpandedNeighborIdx(chunkIndex);
    if (neighborsByIndex[chunkIndex] || !docId) return;
    setNeighborLoadingIdx(chunkIndex);
    try {
      const res = await client.getChunkNeighbors(docId, chunkIndex, new AbortController().signal);
      setNeighborsByIndex((prev) => ({ ...prev, [chunkIndex]: res }));
    } catch {
      // Inline fallback shown in panel.
    } finally {
      setNeighborLoadingIdx((prev) => (prev === chunkIndex ? undefined : prev));
    }
  }

  async function handleOpenPdf() {
    if (!data) return;
    setOpenNotice('');
    try {
      await openTextbookOriginal(data.content_sha256, data.textbook_name);
    } catch (err) {
      setOpenNotice(
        err instanceof Error ? err.message : '无法调用系统默认程序打开原件（请在桌面应用中打开）。',
      );
    }
  }

  const totalPages = data ? Math.max(1, Math.ceil(data.filtered_total / PAGE_SIZE)) : 1;
  const currentPageNum = Math.floor(offset / PAGE_SIZE) + 1;

  return (
    <section className="chunk-browser" aria-label="Chunk 原文浏览器">
      <div className="chunk-browser-head">
        <div>
          <h3>Chunk 原文浏览器</h3>
          <p className="library-hint">
            按页码、章节或关键词检查教材解析与切片质量，核对字符长度、向量化状态、解析警告及相邻上下文。
          </p>
        </div>
        {data && (
          <button type="button" className="secondary-button" onClick={() => void handleOpenPdf()}>
            打开原文件
          </button>
        )}
      </div>

      {openNotice && <p role="status" className="lab-degrade-box">{openNotice}</p>}

      <div className="chunk-browser-toolbar">
        <label className="lab-control chunk-doc-select">
          <span>选择教材</span>
          <select
            aria-label="选择要浏览的教材"
            value={docId}
            onChange={(e) => handleDocChange(e.target.value)}
          >
            {documents.map((doc) => (
              <option key={doc.document_id} value={doc.document_id}>
                {doc.name}（{doc.chunk_count ?? 0} 切片）
              </option>
            ))}
          </select>
        </label>

        <label className="lab-control">
          <span>按页码筛选</span>
          <select
            aria-label="按页码筛选"
            value={pageFilter ?? ''}
            onChange={(e) => {
              const val = e.target.value;
              setPageFilter(val ? Number(val) : undefined);
              setOffset(0);
            }}
          >
            <option value="">全部页码（{data?.pages.length ?? 0} 页）</option>
            {(data?.pages ?? []).map((p) => (
              <option key={p} value={p}>第 {p} 页</option>
            ))}
          </select>
        </label>

        <label className="lab-control">
          <span>按章节筛选</span>
          <select
            aria-label="按章节筛选"
            value={sectionFilter}
            onChange={(e) => {
              setSectionFilter(e.target.value);
              setOffset(0);
            }}
          >
            <option value="">全部章节（{data?.sections.length ?? 0}）</option>
            {(data?.sections ?? []).map((sec, idx) => (
              <option key={`${sec.title}-${idx}`} value={sec.title}>
                {sec.title}（{sec.chunk_count} 条{sec.page_start ? `, p.${sec.page_start}` : ''}）
              </option>
            ))}
          </select>
        </label>

        <form className="chunk-search-form" onSubmit={handleKeywordSubmit}>
          <input
            type="search"
            aria-label="切片内关键词搜索"
            placeholder="搜索切片原文…"
            value={keywordInput}
            onChange={(e) => setKeywordInput(e.target.value)}
          />
          <button type="submit" className="secondary-button">筛选</button>
          {(pageFilter !== undefined || sectionFilter || appliedKeyword) && (
            <button type="button" className="secondary-button" onClick={handleResetFilters}>
              重置
            </button>
          )}
        </form>
      </div>

      {data && (
        <div className="chunk-meta-strip">
          <span>总切片：<strong>{data.total_chunks}</strong></span>
          <span>当前筛选：<strong>{data.filtered_total}</strong></span>
          <span>解析器版本：<code>v{data.parser_version ?? '—'}</code></span>
          <span>切片器版本：<code>v{data.chunker_version ?? '—'}</code></span>
          {(!data.parse_spec_current || !data.chunker_current) ? (
            <span className="index-status-pill tone-keyword">旧版切片 · 建议重建升级</span>
          ) : (
            <span className="index-status-pill tone-ready">P10.4 最新版解析与切片</span>
          )}
        </div>
      )}

      {error && <p role="alert" className="lab-alert-error">{error}</p>}
      {loading && <p role="status">正在加载教材切片…</p>}

      {!loading && data && data.chunks.length === 0 && (
        <p className="search-empty">当前筛选条件下没有匹配的切片。</p>
      )}

      {!loading && data && data.chunks.length > 0 && (
        <>
          <ol className="chunk-card-list" aria-label="教材 Chunk 列表">
            {data.chunks.map((chunk) => {
              const isNeighborOpen = expandedNeighborIdx === chunk.chunk_index;
              const neighborData = neighborsByIndex[chunk.chunk_index];
              const isLoadingNeighbor = neighborLoadingIdx === chunk.chunk_index;
              return (
                <li key={chunk.chunk_id} className="chunk-item-card">
                  <div className="chunk-item-head">
                    <span className="badge badge-chunk">Chunk #{chunk.chunk_index}</span>
                    <span className="badge badge-location">{chunk.location_summary}</span>
                    {chunk.section_path.length > 0 && (
                      <span className="chunk-section-crumb">{chunk.section_path.join(' / ')}</span>
                    )}
                    <span className="chunk-char-count">{chunk.char_count} 字符</span>
                    <span className={`index-status-pill ${chunk.embedded ? 'tone-ready' : 'tone-pending'}`}>
                      {chunk.embedded ? '已向量化' : '未向量化'}
                    </span>
                    {chunk.warnings.length > 0 && (
                      <span className="badge badge-warning" title={chunk.warnings.map((w) => w.message).join('；')}>
                        ⚠️ {chunk.warnings.length} 条解析警告
                      </span>
                    )}
                    {(chunk.has_prev || chunk.has_next) && (
                      <button
                        type="button"
                        className="secondary-button small chunk-neighbor-btn"
                        aria-expanded={isNeighborOpen}
                        onClick={() => void handleToggleNeighbors(chunk.chunk_index)}
                      >
                        {isNeighborOpen ? '收起相邻切片' : '查看相邻 Chunk'}
                      </button>
                    )}
                  </div>

                  {chunk.warnings.length > 0 && (
                    <div className="evidence-warnings">
                      解析警告：{chunk.warnings.map((w) => `${w.code}${w.physical_page ? `(p.${w.physical_page})` : ''}: ${w.message}`).join('；')}
                    </div>
                  )}

                  <pre className="evidence-snippet chunk-text-pre">{chunk.text}</pre>

                  {isNeighborOpen && (
                    <div className="chunk-neighbors-panel" aria-label={`Chunk #${chunk.chunk_index} 相邻上下文`}>
                      {isLoadingNeighbor ? (
                        <p role="status">正在读取相邻切片…</p>
                      ) : !neighborData ? (
                        <p className="lab-muted">无法读取相邻切片。</p>
                      ) : (
                        <div className="chunk-neighbors-grid">
                          <div className="chunk-neighbor-col">
                            <h5>前一个切片（{neighborData.previous ? `#${neighborData.previous.chunk_index} · ${neighborData.previous.location_summary}` : '无'}）</h5>
                            {neighborData.previous ? (
                              <pre className="evidence-snippet neighbor-pre">{neighborData.previous.text}</pre>
                            ) : (
                              <p className="lab-muted">已是全书首条切片。</p>
                            )}
                          </div>
                          <div className="chunk-neighbor-col">
                            <h5>后一个切片（{neighborData.next ? `#${neighborData.next.chunk_index} · ${neighborData.next.location_summary}` : '无'}）</h5>
                            {neighborData.next ? (
                              <pre className="evidence-snippet neighbor-pre">{neighborData.next.text}</pre>
                            ) : (
                              <p className="lab-muted">已是全书末条切片。</p>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ol>

          <div className="chunk-pagination">
            <button
              type="button"
              className="secondary-button"
              disabled={offset === 0}
              onClick={() => setOffset((prev) => Math.max(0, prev - PAGE_SIZE))}
            >
              上一页
            </button>
            <span>
              第 <strong>{currentPageNum}</strong> / {totalPages} 页（共 {data.filtered_total} 条）
            </span>
            <button
              type="button"
              className="secondary-button"
              disabled={offset + PAGE_SIZE >= data.filtered_total}
              onClick={() => setOffset((prev) => prev + PAGE_SIZE)}
            >
              下一页
            </button>
          </div>
        </>
      )}
    </section>
  );
}
