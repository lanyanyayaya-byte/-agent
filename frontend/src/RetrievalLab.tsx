import { useState, type FormEvent } from 'react';
import {
  ApiRequestError,
  type ApiClient,
  type CandidateTraceView,
  type DocumentIndexStatus,
  type KnowledgeSearchResponse,
  type RetrievalMode,
  type RetrievalStrategy,
  type RewriteMode,
} from './api/apiClient';

type LabModeOption = 'hybrid_rerank' | 'hybrid' | 'vector' | 'keyword';
type CandidateFilter = 'all' | 'selected' | 'excluded';

const FORMAT_LABELS: Record<'pdf' | 'docx' | 'md', string> = {
  pdf: 'PDF',
  docx: 'DOCX',
  md: 'Markdown',
};

const STRATEGY_LABELS: Record<string, string> = {
  global: '全局相关性检索',
  scoped: '限定教材检索',
  compare: '跨教材对比',
  book_summary: '全文 / 章节总结',
  auto: '自动策略',
};

const STAGE_LABELS: Record<string, string> = {
  query_rewrite: '查询改写',
  scope_selection: '教材范围筛选',
  bm25: 'BM25 关键词召回',
  vector: '向量语义召回',
  rrf_fusion: 'RRF 融合排序',
  rerank: 'BGE-Reranker 精排',
  context_expansion: '相邻 Chunk 扩展',
  book_summary: '全文/章节覆盖规划',
};

const STAGE_STATUS_LABELS: Record<string, string> = {
  ok: '完成',
  skipped: '跳过',
  degraded: '降级',
  failed: '失败',
};

const EXCLUSION_LABELS: Record<string, string> = {
  not_in_top_k: '未进入 Top-K',
  topk_cutoff: '未进入 Top-K',
  below_min_similarity: '低于相似度阈值',
  per_doc_cap: '单本配额上限',
  deduplicated: '去重过滤',
  reference_demoted: '参考文献降权',
};

function formatScore(val?: number | null, digits = 3): string {
  if (val === undefined || val === null || Number.isNaN(val)) return '—';
  return val.toFixed(digits);
}

function formatRank(rank?: number | null): string {
  if (rank === undefined || rank === null) return '—';
  return `#${rank}`;
}

export default function RetrievalLab({
  client,
  documents,
  defaultRerank,
  onInspectChunk,
}: {
  client: ApiClient;
  documents: DocumentIndexStatus[];
  defaultRerank: boolean;
  onInspectChunk?: (documentId: string, page?: number, query?: string) => void;
}) {
  const [query, setQuery] = useState('');
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [strategy, setStrategy] = useState<RetrievalStrategy>('global');
  const [labMode, setLabMode] = useState<LabModeOption>(defaultRerank ? 'hybrid_rerank' : 'hybrid');
  const [rewriteMode, setRewriteMode] = useState<RewriteMode>('rules');
  const [topK, setTopK] = useState(8);
  const [candidateK, setCandidateK] = useState(40);
  const [minSimilarity, setMinSimilarity] = useState(0.3);
  const [expandNeighbors, setExpandNeighbors] = useState(true);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState('');
  const [result, setResult] = useState<KnowledgeSearchResponse>();
  const [candidateFilter, setCandidateFilter] = useState<CandidateFilter>('all');
  const [expandedCandidateId, setExpandedCandidateId] = useState<string>();
  const [showSkippedDocs, setShowSkippedDocs] = useState(false);

  function toggleDoc(docId: string) {
    setSelectedDocs((prev) => {
      if (strategy === 'book_summary') {
        return prev.includes(docId) ? [] : [docId];
      }
      return prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId];
    });
  }

  function handleStrategyChange(next: RetrievalStrategy) {
    setStrategy(next);
    if (next === 'book_summary' && selectedDocs.length > 1) {
      setSelectedDocs([selectedDocs[0]]);
    }
    if (next === 'compare' && topK < 12) {
      setTopK(12);
    }
  }

  async function handleRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || searching) return;
    setSearching(true);
    setSearchError('');
    setExpandedCandidateId(undefined);

    const mode: RetrievalMode = labMode === 'hybrid_rerank' ? 'hybrid' : labMode;
    const rerank = labMode === 'hybrid_rerank';
    const effectiveStrategy: RetrievalStrategy =
      strategy === 'global' && selectedDocs.length > 0 ? 'scoped' : strategy;

    try {
      const res = await client.debugSearchKnowledge(
        {
          query: trimmed,
          document_ids: selectedDocs.length > 0 ? selectedDocs : null,
          strategy: effectiveStrategy,
          mode,
          coverage_mode: effectiveStrategy === 'compare' ? 'per_document' : 'global',
          rerank,
          rewrite: rewriteMode,
          top_k: topK,
          candidate_k: candidateK,
          min_vector_similarity: minSimilarity,
          expand_neighbors: expandNeighbors,
        },
        new AbortController().signal,
      );
      setResult(res);
    } catch (error) {
      setSearchError(
        error instanceof ApiRequestError
          ? `${error.message}${error.code ? ` (${error.code})` : ''}`
          : '检索实验室执行失败，请检查后端连接与教材索引状态。',
      );
    } finally {
      setSearching(false);
    }
  }

  const trace = result?.trace ?? null;
  const allCandidates: CandidateTraceView[] = trace?.candidates ?? [];
  const selectedCandidatesCount = allCandidates.filter((c) => c.selected).length;
  const excludedCandidatesCount = allCandidates.length - selectedCandidatesCount;
  const filteredCandidates = allCandidates.filter((c) => {
    if (candidateFilter === 'selected') return c.selected;
    if (candidateFilter === 'excluded') return !c.selected;
    return true;
  });

  const participatedDocs = (trace?.documents ?? []).filter((d) => !d.skip_reason);
  const skippedDocs = (trace?.documents ?? []).filter((d) => Boolean(d.skip_reason));

  return (
    <section className="retrieval-lab" aria-label="RAG 检索实验室">
      <div className="lab-header">
        <div>
          <h3>RAG 检索实验室</h3>
          <p className="library-hint">
            直接复用生产检索链路（不生成最终问答）。关闭 LLM 改写时完全独立于 DeepSeek 运行；所有分数均为检索相关度指标，不代表回答准确率。
          </p>
        </div>
      </div>

      <form className="lab-form" onSubmit={(e) => void handleRun(e)}>
        <div className="library-search-bar">
          <input
            type="search"
            aria-label="输入检索实验室问题"
            placeholder="输入中英文问题、定理、术语或总结要求（如：What is the ADI scheme? / 完整总结非结构网格生成方法）"
            maxLength={400}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="submit" disabled={searching || !query.trim()}>
            {searching ? '正在运行检索…' : '运行检索诊断'}
          </button>
        </div>

        <div className="lab-controls-grid">
          <label className="lab-control">
            <span>检索模式</span>
            <select
              aria-label="实验室检索模式"
              value={labMode}
              onChange={(e) => setLabMode(e.target.value as LabModeOption)}
            >
              <option value="hybrid_rerank">Hybrid + Reranker（推荐）</option>
              <option value="hybrid">Hybrid（BM25 + 向量 RRF）</option>
              <option value="vector">仅 Vector（BGE-M3）</option>
              <option value="keyword">仅 BM25（离线关键词）</option>
            </select>
          </label>

          <label className="lab-control">
            <span>检索策略</span>
            <select
              aria-label="实验室检索策略"
              value={strategy}
              onChange={(e) => handleStrategyChange(e.target.value as RetrievalStrategy)}
            >
              <option value="global">全局相关性（Global / Scoped）</option>
              <option value="compare">跨教材对比（Compare）</option>
              <option value="book_summary">单本全文 / 章节总结</option>
            </select>
          </label>

          <label className="lab-control">
            <span>双语查询改写</span>
            <select
              aria-label="双语查询改写模式"
              value={rewriteMode}
              onChange={(e) => setRewriteMode(e.target.value as RewriteMode)}
            >
              <option value="off">关闭（0 LLM 调用）</option>
              <option value="rules">轻量规则词表（离线）</option>
              <option value="llm">DeepSeek 双语改写</option>
            </select>
          </label>

          <label className="lab-control lab-control-num">
            <span>最终 Top-K</span>
            <input
              type="number"
              aria-label="最终 Top-K"
              min={1}
              max={40}
              value={topK}
              onChange={(e) => setTopK(Math.max(1, Math.min(40, Number(e.target.value) || 8)))}
            />
          </label>

          <label className="lab-control lab-control-num">
            <span>候选召回数</span>
            <input
              type="number"
              aria-label="候选召回数"
              min={10}
              max={120}
              step={5}
              value={candidateK}
              onChange={(e) => setCandidateK(Math.max(10, Math.min(120, Number(e.target.value) || 40)))}
            />
          </label>

          <label className="lab-control lab-control-num">
            <span>向量相似度阈值</span>
            <input
              type="number"
              aria-label="向量相似度阈值"
              min={0}
              max={0.95}
              step={0.05}
              value={minSimilarity}
              onChange={(e) => setMinSimilarity(Math.max(0, Math.min(0.95, Number(e.target.value) || 0)))}
            />
          </label>

          <label className="lab-toggle-control">
            <input
              type="checkbox"
              checked={expandNeighbors}
              onChange={(e) => setExpandNeighbors(e.target.checked)}
            />
            <span>相邻 Chunk 上下文扩展</span>
          </label>
        </div>

        {documents.length > 0 && (
          <div className="library-doc-filters" role="group" aria-label="实验室检索教材范围">
            <span className="filter-label">
              {strategy === 'book_summary' ? '选择总结教材（单本）：' : '检索范围：'}
            </span>
            {strategy !== 'book_summary' && (
              <button
                type="button"
                className={`filter-chip ${selectedDocs.length === 0 ? 'active' : ''}`}
                onClick={() => setSelectedDocs([])}
              >
                全部教材（{documents.length}）
              </button>
            )}
            {documents.map((doc) => {
              const isSelected = selectedDocs.includes(doc.document_id);
              return (
                <button
                  key={doc.document_id}
                  type="button"
                  className={`filter-chip ${isSelected ? 'active' : ''}`}
                  aria-pressed={isSelected}
                  onClick={() => toggleDoc(doc.document_id)}
                >
                  <bdi>{doc.name}</bdi>
                </button>
              );
            })}
          </div>
        )}
      </form>

      {searchError && <p role="alert" className="lab-alert-error">{searchError}</p>}

      {result && (
        <div className="lab-results" aria-live="polite">
          {/* Summary KPI Bar */}
          <div className="lab-summary-banner">
            <div className="lab-summary-badges">
              <span className="badge badge-channel">
                {STRATEGY_LABELS[trace?.actual_strategy ?? result.actual_strategy ?? 'global'] ?? result.actual_strategy}
              </span>
              <span className="badge badge-format">
                通道：{trace?.effective_channels?.join(' + ') || result.channel_status}
              </span>
              {result.rerank_used && <span className="badge badge-rerank">BGE-Reranker 已精排</span>}
              {(trace?.context_expansion ?? 0) > 0 && (
                <span className="badge badge-neighbor">相邻扩展 +{trace?.context_expansion}</span>
              )}
              {trace?.degraded && <span className="badge badge-warning">存在降级</span>}
            </div>
            <div className="lab-summary-metrics">
              <span>范围教材：<strong>{trace?.in_scope_documents ?? documents.length}</strong></span>
              <span>可检索：<strong>{trace?.searchable_documents ?? 0}</strong></span>
              <span>命中教材：<strong>{trace?.hit_documents ?? 0}</strong></span>
              <span>最终证据：<strong>{result.hits.length}</strong></span>
              <span>总耗时：<strong>{(trace?.total_elapsed_ms ?? 0).toFixed(1)} ms</strong></span>
            </div>
          </div>

          {/* Degradation / Error banners */}
          {(result.diagnostic || result.rerank_diagnostic || (trace?.degrade_reasons?.length ?? 0) > 0) && (
            <div className="lab-degrade-box" role="status">
              {result.diagnostic && (
                <div>降级提示：{result.diagnostic.message}（<code>{result.diagnostic.code}</code>）</div>
              )}
              {result.rerank_diagnostic && (
                <div>精排降级：{result.rerank_diagnostic.message}（<code>{result.rerank_diagnostic.code}</code>）</div>
              )}
              {trace?.degrade_reasons?.map((reason, idx) => (
                <div key={idx}>Trace 降级记录：{reason}</div>
              ))}
            </div>
          )}

          {/* Query Plan & Bilingual Expansion */}
          {trace && (
            <section className="lab-card" aria-label="查询改写与双语扩展">
              <div className="lab-card-head">
                <h4>1. 查询改写与中英文术语扩展</h4>
                <span className="lab-meta-tag">
                  改写模式：{trace.rewrite_mode || 'off'}
                  {trace.rewrite_reason ? ` · ${trace.rewrite_reason}` : ''}
                </span>
              </div>
              <div className="lab-query-grid">
                <div>
                  <span className="lab-field-label">原始问题</span>
                  <div className="lab-query-value">{trace.original_query}</div>
                </div>
                <div>
                  <span className="lab-field-label">实际主查询</span>
                  <div className="lab-query-value">{trace.resolved_query}</div>
                </div>
              </div>
              <div className="lab-terms-row">
                <span className="lab-field-label">中文术语：</span>
                {(trace.chinese_terms?.length ?? 0) > 0 ? (
                  trace.chinese_terms!.map((term) => (
                    <span key={term} className="term-pill term-zh">{term}</span>
                  ))
                ) : (
                  <span className="lab-muted">无额外中文扩展</span>
                )}
                <span className="lab-field-label">英文术语：</span>
                {(trace.english_terms?.length ?? 0) > 0 ? (
                  trace.english_terms!.map((term) => (
                    <span key={term} className="term-pill term-en">{term}</span>
                  ))
                ) : (
                  <span className="lab-muted">无额外英文扩展</span>
                )}
              </div>
              {trace.query_variants.length > 0 && (
                <div className="lab-variants-list">
                  <span className="lab-field-label">实际检索变体（按优先级）：</span>
                  <ol>
                    {trace.query_variants.map((v, idx) => (
                      <li key={idx}>
                        <code>[{idx}]</code> {v}
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </section>
          )}

          {/* Book Summary Coverage Plan */}
          {trace?.summary_coverage && (
            <section className="lab-card lab-summary-coverage-card" aria-label="全文与章节覆盖规划">
              <div className="lab-card-head">
                <h4>全文 / 章节总结覆盖范围</h4>
                <span className="badge badge-channel">{trace.summary_coverage.summary_mode}</span>
              </div>
              <p className="lab-coverage-line">
                教材：《<strong>{trace.summary_coverage.textbook_name}</strong>》 · 选中切片{' '}
                <strong>{trace.summary_coverage.selected_chunks}</strong> / {trace.summary_coverage.total_chunks} · 覆盖页码：
                <strong>{trace.summary_coverage.covered_page_range || '—'}</strong>（覆盖{' '}
                {trace.summary_coverage.covered_pages_count ?? 0} 页，未覆盖{' '}
                {trace.summary_coverage.uncovered_pages_count ?? 0} 页）
              </p>
              {(trace.summary_coverage.covered_sections?.length ?? 0) > 0 && (
                <div className="lab-section-tags">
                  <span className="lab-field-label">已覆盖章节：</span>
                  {trace.summary_coverage.covered_sections!.map((sec, i) => (
                    <span key={i} className="term-pill term-zh">{sec}</span>
                  ))}
                </div>
              )}
              {(trace.summary_coverage.skipped_sections?.length ?? 0) > 0 && (
                <div className="lab-section-tags">
                  <span className="lab-field-label">已跳过章节：</span>
                  {trace.summary_coverage.skipped_sections!.map((sec, i) => (
                    <span key={i} className="term-pill term-skipped" title={sec.reason}>
                      {sec.section}（{sec.reason}）
                    </span>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Stage Pipeline Timings */}
          {trace && trace.stages.length > 0 && (
            <section className="lab-card" aria-label="各阶段耗时与候选数">
              <h4>2. 检索链路各阶段耗时</h4>
              <div className="lab-stages-row">
                {trace.stages.map((st, idx) => (
                  <div key={`${st.stage}-${idx}`} className={`lab-stage-pill status-${st.status}`}>
                    <div className="lab-stage-top">
                      <strong>{STAGE_LABELS[st.stage] ?? st.stage}</strong>
                      <span className={`stage-status-dot status-${st.status}`}>
                        {STAGE_STATUS_LABELS[st.status] ?? st.status}
                      </span>
                    </div>
                    <div className="lab-stage-bottom">
                      <span>候选 {st.candidate_count}</span>
                      <span>{st.elapsed_ms.toFixed(1)} ms</span>
                    </div>
                    {st.message && <div className="lab-stage-msg" title={st.message}>{st.message}</div>}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Participating vs Skipped Textbooks */}
          {trace && trace.documents.length > 0 && (
            <section className="lab-card" aria-label="参与与未参与检索的教材">
              <div className="lab-card-head">
                <h4>
                  3. 教材参与状态（参与 {participatedDocs.length} 本 / 未参与 {skippedDocs.length} 本）
                </h4>
                {skippedDocs.length > 0 && (
                  <button
                    type="button"
                    className="secondary-button small"
                    onClick={() => setShowSkippedDocs((prev) => !prev)}
                  >
                    {showSkippedDocs ? '隐藏未参与教材' : `查看未参与教材（${skippedDocs.length}）`}
                  </button>
                )}
              </div>
              <div className="lab-doc-grid">
                {participatedDocs.slice(0, 18).map((doc) => (
                  <div
                    key={doc.document_id}
                    className={`lab-doc-chip ${doc.returned_count > 0 ? 'has-hits' : ''}`}
                  >
                    <strong title={doc.textbook_name ?? doc.document_id}>
                      <bdi>{doc.textbook_name ?? doc.document_id}</bdi>
                    </strong>
                    <span>
                      {doc.searched_channels.join('+')} · 命中 {doc.returned_count} 条
                    </span>
                  </div>
                ))}
              </div>
              {showSkippedDocs && skippedDocs.length > 0 && (
                <div className="lab-skipped-list">
                  {skippedDocs.map((doc) => (
                    <div key={doc.document_id} className="lab-skipped-item">
                      <strong><bdi>{doc.textbook_name ?? doc.document_id}</bdi></strong>
                      <span className="lab-muted">未参与原因：{doc.skip_reason}</span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}

          {/* Candidate Stage Comparison & Filtered Reasons */}
          {allCandidates.length > 0 && (
            <section className="lab-card" aria-label="候选排名变化与过滤原因">
              <div className="lab-card-head">
                <h4>4. 候选 Chunk 各阶段排名与过滤诊断（共记录 {allCandidates.length} 条）</h4>
                <div className="lab-filter-tabs" role="tablist">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={candidateFilter === 'all'}
                    className={`lab-tab-btn ${candidateFilter === 'all' ? 'active' : ''}`}
                    onClick={() => setCandidateFilter('all')}
                  >
                    全部 ({allCandidates.length})
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={candidateFilter === 'selected'}
                    className={`lab-tab-btn ${candidateFilter === 'selected' ? 'active' : ''}`}
                    onClick={() => setCandidateFilter('selected')}
                  >
                    最终入围 ({selectedCandidatesCount})
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={candidateFilter === 'excluded'}
                    className={`lab-tab-btn ${candidateFilter === 'excluded' ? 'active' : ''}`}
                    onClick={() => setCandidateFilter('excluded')}
                  >
                    被过滤 ({excludedCandidatesCount})
                  </button>
                </div>
              </div>
              <p className="lab-score-disclaimer">
                注：余弦相似度与 Reranker 分数仅表示语义/相关性排序强弱，不代表最终回答准确率。
              </p>
              <div className="library-table lab-candidates-table-wrap">
                <table aria-label="候选 Chunk 阶段排名对比">
                  <thead>
                    <tr>
                      <th scope="col">教材 / 位置 / Chunk</th>
                      <th scope="col">BM25</th>
                      <th scope="col">Vector (余弦)</th>
                      <th scope="col">RRF 融合</th>
                      <th scope="col">Reranker</th>
                      <th scope="col">状态 / 过滤原因</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredCandidates.map((cand, idx) => {
                      const candKey = `${cand.chunk_id}-${cand.is_neighbor_expansion ? 'nb' : 'core'}-${idx}`;
                      const isExpanded = expandedCandidateId === candKey;
                      return (
                        <tr
                          key={candKey}
                          className={`cand-row ${cand.selected ? 'is-selected' : 'is-excluded'}`}
                          onClick={() => setExpandedCandidateId(isExpanded ? undefined : candKey)}
                        >
                          <td>
                            <div className="cand-title-cell">
                              <strong><bdi>{cand.textbook_name ?? cand.document_id}</bdi></strong>
                              <span className="cand-sub">
                                #{cand.chunk_index}
                                {cand.page_range ? ` · ${cand.page_range}` : ''}
                                {cand.section_path.length > 0 ? ` · ${cand.section_path.join(' / ')}` : ''}
                              </span>
                              <div className="cand-flags">
                                {cand.is_neighbor_expansion && <span className="mini-badge neighbor">相邻扩展</span>}
                                {cand.is_reference_chunk && <span className="mini-badge ref">参考文献降权</span>}
                              </div>
                              {isExpanded && cand.snippet && (
                                <pre className="cand-snippet-inline">{cand.snippet}</pre>
                              )}
                            </div>
                          </td>
                          <td className="num-cell">
                            {formatRank(cand.bm25_rank)}
                            {cand.bm25_raw != null && <small> ({formatScore(cand.bm25_raw, 2)})</small>}
                          </td>
                          <td className="num-cell">
                            {formatRank(cand.vector_rank)}
                            {cand.cosine != null && <small> ({formatScore(cand.cosine, 3)})</small>}
                          </td>
                          <td className="num-cell">
                            {formatRank(cand.rrf_rank)}
                            {cand.rrf_score != null && <small> ({formatScore(cand.rrf_score, 4)})</small>}
                          </td>
                          <td className="num-cell">
                            {formatRank(cand.rerank_rank)}
                            {cand.rerank_score != null && <small> ({formatScore(cand.rerank_score, 3)})</small>}
                          </td>
                          <td>
                            {cand.selected ? (
                              <span className="index-status-pill tone-ready">已入围</span>
                            ) : (
                              <span className="index-status-pill tone-muted">
                                {EXCLUSION_LABELS[cand.exclusion_reason ?? ''] ?? cand.exclusion_reason ?? '已过滤'}
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Final Verified Evidence */}
          <section className="lab-card" aria-label="最终入选证据">
            <h4>5. 最终入选证据（{result.hits.length} 条）</h4>
            {result.hits.length === 0 ? (
              <p className="search-empty">未检索到满足条件的教材证据。可检查上方被过滤候选、调低相似度阈值或确认向量索引状态。</p>
            ) : (
              <ol className="evidence-list" aria-label="检索实验室最终证据列表">
                {result.hits.map((hit, idx) => (
                  <li key={`${hit.evidence_id}-${idx}`} className="evidence-card">
                    <div className="evidence-header">
                      <span className="evidence-index">证据 #{idx + 1}</span>
                      <strong className="evidence-book"><bdi>{hit.textbook_name}</bdi></strong>
                      <span className="badge badge-format">{FORMAT_LABELS[hit.format]}</span>
                      <span className="badge badge-location">{hit.location_summary}</span>
                      <span className="badge badge-chunk">Chunk #{hit.chunk_index}</span>
                      {hit.is_neighbor_expansion && <span className="badge badge-neighbor">相邻扩展</span>}
                      {hit.rerank_score != null && (
                        <span className="badge badge-rerank">Rerank {hit.rerank_score.toFixed(3)}</span>
                      )}
                      <span className="evidence-score">RRF {hit.fusion_score.toFixed(4)}</span>
                      {onInspectChunk && (
                        <button
                          type="button"
                          className="secondary-button small"
                          onClick={() => onInspectChunk(hit.document_id, hit.physical_pages[0])}
                        >
                          在 Chunk 浏览器查看
                        </button>
                      )}
                    </div>
                    {hit.section_path.length > 0 && (
                      <div className="evidence-section">章节：{hit.section_path.join(' / ')}</div>
                    )}
                    {hit.warnings.length > 0 && (
                      <div className="evidence-warnings" role="note">
                        ⚠️ 解析提示：{hit.warnings.map((w) => `${w.code}（${w.message}）`).join('；')}
                      </div>
                    )}
                    <pre className="evidence-snippet">{hit.body_text}</pre>
                    <div className="evidence-footer">
                      {hit.channel_ranks.map((r) => (
                        <span key={r.channel} className="rank-pill">
                          {r.channel === 'keyword' ? 'BM25' : 'Vector'} #{r.rank}（{r.score.toFixed(3)}）
                        </span>
                      ))}
                      <span className="evidence-id-tag" title={hit.chunk_id}>
                        {hit.chunk_id}
                      </span>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </div>
      )}
    </section>
  );
}
