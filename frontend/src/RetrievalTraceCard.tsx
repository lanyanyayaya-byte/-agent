import { useState } from 'react';
import type { ApiClient, RetrievalTraceData, RetrievalTraceSummary } from './api/apiClient';

const STRATEGY_LABELS: Record<string, string> = {
  global: '全局检索',
  scoped: '限定教材',
  compare: '跨教材对比',
  book_summary: '全文/章节总结',
  auto: '自动策略',
};

const STAGE_LABELS: Record<string, string> = {
  query_rewrite: '查询改写',
  scope_selection: '教材筛选',
  bm25: 'BM25',
  vector: '向量',
  rrf_fusion: 'RRF 融合',
  rerank: 'Reranker',
  context_expansion: '相邻扩展',
  book_summary: '章节规划',
};

const EXCLUSION_LABELS: Record<string, string> = {
  not_in_top_k: '未进 Top-K',
  topk_cutoff: '未进 Top-K',
  below_min_similarity: '低于阈值',
  per_doc_cap: '单本上限',
  deduplicated: '去重',
  reference_demoted: '参考文献降权',
};

function fmtScore(val?: number | null, digits = 3): string {
  if (val === undefined || val === null || Number.isNaN(val)) return '—';
  return val.toFixed(digits);
}

export default function RetrievalTraceCard({
  client,
  runId,
  summary,
  liveTrace,
  isLegacyRagRun = false,
}: {
  client?: ApiClient;
  runId: string;
  summary?: RetrievalTraceSummary | null;
  liveTrace?: RetrievalTraceData | null;
  isLegacyRagRun?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const [loadedTrace, setLoadedTrace] = useState<RetrievalTraceData | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadErr, setLoadErr] = useState('');

  const effectiveTrace = liveTrace ?? loadedTrace;

  if (!summary && !effectiveTrace) {
    if (isLegacyRagRun) {
      return (
        <div className="trace-card-legacy" aria-label="检索过程记录">
          <span>🔍 检索过程：历史版本未记录</span>
        </div>
      );
    }
    return null;
  }

  const effectiveSummary: RetrievalTraceSummary = summary ?? {
    trace_id: effectiveTrace!.trace_id,
    actual_strategy: effectiveTrace!.actual_strategy,
    requested_mode: effectiveTrace!.requested_mode,
    effective_channels: effectiveTrace!.effective_channels,
    in_scope_documents: effectiveTrace!.in_scope_documents,
    searchable_documents: effectiveTrace!.searchable_documents,
    hit_documents: effectiveTrace!.hit_documents,
    cited_documents: effectiveTrace!.cited_documents,
    prompt_evidence_count: effectiveTrace!.prompt_evidence_count,
    bm25_used: Boolean(effectiveTrace!.bm25_used),
    vector_used: Boolean(effectiveTrace!.vector_used),
    rerank_used: Boolean(effectiveTrace!.rerank_used),
    total_elapsed_ms: effectiveTrace!.total_elapsed_ms ?? 0,
    context_expansion: effectiveTrace!.context_expansion ?? 0,
    degraded: effectiveTrace!.degraded,
    degrade_reasons: effectiveTrace!.degrade_reasons,
    warnings: effectiveTrace!.warnings,
    error_code: effectiveTrace!.error_code ?? null,
    error_message: effectiveTrace!.error_message ?? null,
    summary_coverage: effectiveTrace!.summary_coverage ?? null,
  };

  async function handleToggle() {
    const nextOpen = !expanded;
    setExpanded(nextOpen);
    if (nextOpen && !effectiveTrace && client && !loading) {
      setLoading(true);
      setLoadErr('');
      try {
        const res = await client.getRunRetrievalTrace(runId, new AbortController().signal);
        setLoadedTrace(res.trace);
        if (!res.trace) {
          setLoadErr('未找到该次运行的详细检索记录。');
        }
      } catch {
        setLoadErr('读取检索过程详情失败。');
      } finally {
        setLoading(false);
      }
    }
  }

  const channelTokens: string[] = [];
  if (effectiveSummary.bm25_used) channelTokens.push('BM25');
  if (effectiveSummary.vector_used) channelTokens.push('向量');
  if (effectiveSummary.rerank_used) channelTokens.push('Reranker');
  const channelLabel = channelTokens.length > 0
    ? channelTokens.join(' + ')
    : (effectiveSummary.effective_channels.join(' + ') || effectiveSummary.requested_mode);

  const hasError = Boolean(effectiveSummary.error_code);

  return (
    <section
      className={`retrieval-trace-card ${hasError ? 'is-error' : effectiveSummary.degraded ? 'is-degraded' : ''}`}
      aria-label="本次教材检索过程"
    >
      <button
        type="button"
        className="retrieval-trace-toggle"
        aria-expanded={expanded}
        onClick={() => void handleToggle()}
      >
        <span className="trace-caret">{expanded ? '▾' : '▸'}</span>
        <span className="trace-badge-strategy">
          {STRATEGY_LABELS[effectiveSummary.actual_strategy] ?? effectiveSummary.actual_strategy}
        </span>
        <span className="trace-summary-text">
          {channelLabel} · 检索 {effectiveSummary.searchable_documents}/{effectiveSummary.in_scope_documents} 本 · 命中{' '}
          {effectiveSummary.hit_documents} 本 · {effectiveSummary.prompt_evidence_count} 条证据
          {effectiveSummary.context_expansion > 0 ? `（含相邻扩展 +${effectiveSummary.context_expansion}）` : ''}
          {effectiveSummary.total_elapsed_ms > 0 ? ` · ${Math.round(effectiveSummary.total_elapsed_ms)}ms` : ''}
        </span>
        {hasError && <span className="trace-status-pill error">检索失败</span>}
        {!hasError && effectiveSummary.degraded && <span className="trace-status-pill warn">已降级</span>}
      </button>

      {expanded && (
        <div className="retrieval-trace-body">
          {loading && <p className="trace-loading">正在加载完整检索轨迹…</p>}
          {loadErr && <p className="trace-error-note">{loadErr}</p>}

          {effectiveSummary.error_message && (
            <div className="trace-banner error">
              检索失败：{effectiveSummary.error_message}
              {effectiveSummary.error_code ? ` (${effectiveSummary.error_code})` : ''}
            </div>
          )}

          {effectiveSummary.degrade_reasons.length > 0 && (
            <div className="trace-banner warn">
              降级原因：{effectiveSummary.degrade_reasons.join('；')}
            </div>
          )}

          {effectiveTrace && (
            <>
              <div className="trace-query-box">
                <div>
                  <strong>原始问题：</strong>
                  <span>{effectiveTrace.original_query}</span>
                </div>
                {effectiveTrace.resolved_query !== effectiveTrace.original_query && (
                  <div>
                    <strong>改写查询：</strong>
                    <span>{effectiveTrace.resolved_query}</span>
                  </div>
                )}
                {effectiveTrace.query_variants.length > 1 && (
                  <div className="trace-variants-row">
                    <strong>检索变体：</strong>
                    {effectiveTrace.query_variants.map((v, i) => (
                      <span key={i} className="trace-variant-chip">{v}</span>
                    ))}
                  </div>
                )}
              </div>

              {effectiveTrace.summary_coverage && (
                <div className="trace-coverage-box">
                  <strong>全文/章节覆盖：</strong>
                  《{effectiveTrace.summary_coverage.textbook_name}》 ·{' '}
                  {effectiveTrace.summary_coverage.summary_mode} · 选中{' '}
                  {effectiveTrace.summary_coverage.selected_chunks}/{effectiveTrace.summary_coverage.total_chunks} 切片 · 页码{' '}
                  {effectiveTrace.summary_coverage.covered_page_range || '—'}
                </div>
              )}

              {effectiveTrace.stages.length > 0 && (
                <div className="trace-stages-strip">
                  {effectiveTrace.stages.map((st, idx) => (
                    <span key={`${st.stage}-${idx}`} className={`trace-stage-chip status-${st.status}`}>
                      {STAGE_LABELS[st.stage] ?? st.stage}: {st.candidate_count} ({st.elapsed_ms.toFixed(0)}ms)
                    </span>
                  ))}
                </div>
              )}

              {effectiveTrace.candidates.length > 0 && (
                <div className="trace-candidates-wrap">
                  <div className="trace-candidates-head">
                    <strong>候选排序与过滤轨迹（前 {Math.min(effectiveTrace.candidates.length, 16)} 条）</strong>
                    <small>注：余弦与 Rerank 为检索相关度分数，不代表回答准确率</small>
                  </div>
                  <table className="trace-candidates-table">
                    <thead>
                      <tr>
                        <th>教材 / 页码</th>
                        <th>BM25</th>
                        <th>Vector</th>
                        <th>RRF</th>
                        <th>Rerank</th>
                        <th>结果</th>
                      </tr>
                    </thead>
                    <tbody>
                      {effectiveTrace.candidates.slice(0, 16).map((c, idx) => (
                        <tr key={`${c.chunk_id}-${idx}`} className={c.selected ? 'is-selected' : 'is-excluded'}>
                          <td>
                            <span className="trace-cand-book">{c.textbook_name ?? c.document_id}</span>
                            <span className="trace-cand-pos">
                              {' '}#{c.chunk_index}{c.page_range ? ` · ${c.page_range}` : ''}
                              {c.is_neighbor_expansion ? ' [相邻扩展]' : ''}
                              {c.is_reference_chunk ? ' [参考文献]' : ''}
                            </span>
                          </td>
                          <td>{c.bm25_rank ? `#${c.bm25_rank}` : '—'}</td>
                          <td>
                            {c.vector_rank ? `#${c.vector_rank}` : '—'}
                            {c.cosine != null ? ` (${fmtScore(c.cosine, 2)})` : ''}
                          </td>
                          <td>{c.rrf_score != null ? fmtScore(c.rrf_score, 4) : '—'}</td>
                          <td>{c.rerank_score != null ? fmtScore(c.rerank_score, 3) : '—'}</td>
                          <td>
                            {c.selected ? (
                              <span className="trace-pill-in">入围</span>
                            ) : (
                              <span className="trace-pill-out">
                                {EXCLUSION_LABELS[c.exclusion_reason ?? ''] ?? c.exclusion_reason ?? '过滤'}
                              </span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}
