// @vitest-environment jsdom

import { invoke } from '@tauri-apps/api/core';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, expect, test, vi } from 'vitest';
import LibraryPanel from './LibraryPanel';
import {
  connectApiClient,
  type DocumentDetails,
  type DocumentIndex,
  type DocumentIndexStatus,
  type IndexJobState,
  type KnowledgeIndexOverview,
  type KnowledgeSettings,
} from './api/apiClient';

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn() }));

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  vi.unstubAllGlobals();
});

const id = 'a'.repeat(32);
const importId = 'bf644ca9-bd5f-4cd9-af10-15603dbccf1e';
const date = '2026-11-22T08:30:00Z';
const original = {
  document_id: id, name: '教材.md', format: 'md', management_status: 'imported',
  created_at: date, updated_at: date, version_count: 1, latest_size_bytes: 9,
  versions: [{ document_version_id: 'b'.repeat(32), size_bytes: 9, content_sha256: 'c'.repeat(64), created_at: date }],
} satisfies DocumentDetails;
const documentsPath = '/api/v1/knowledge/documents';
const importer = vi.mocked(invoke);
const versionId = original.versions[0].document_version_id;
const jobPath = `${documentsPath}/${id}/versions/${versionId}/index-jobs`;
const queued: IndexJobState = {
  job_id: 'e'.repeat(32), document_id: id, document_version_id: versionId, status: 'queued', stage: 'parsing',
  completed_work: 0, total_work: null, error_code: null, retry_count: 0, next_retry_at: null, cancel_requested_at: null,
};
const indexing: IndexJobState = { ...queued, status: 'running', stage: 'embedding', completed_work: 1, total_work: 3 };

const defaultSettings: KnowledgeSettings = {
  auto_index_on_import: true,
  cloud_embedding_enabled: false,
  cloud_rerank_enabled: true,
  enable_bilingual_rewrite: true,
  default_top_k: 8,
  compare_top_k: 12,
  rerank_candidates: 24,
  min_vector_similarity: 0.25,
};

function textbookUnavailable() {
  return Response.json({ error: { code: 'KNOWLEDGE_STORE_UNAVAILABLE', message: 'internal detail' } }, { status: 503 });
}

function toDocStatus(
  doc: DocumentDetails,
  index: Pick<DocumentIndex, 'job' | 'published_readiness'>,
  overrides: Partial<DocumentIndexStatus> = {},
): DocumentIndexStatus {
  const readyHybrid = index.published_readiness === 'ready_hybrid';
  const readyKeyword = index.published_readiness === 'ready_keyword';
  return {
    document_id: doc.document_id,
    name: doc.name,
    format: doc.format,
    version_id: doc.versions.at(-1)?.document_version_id ?? versionId,
    version_count: doc.version_count,
    latest_size_bytes: doc.latest_size_bytes,
    created_at: doc.created_at,
    updated_at: doc.updated_at,
    parse_status: readyHybrid || readyKeyword ? 'ready' : 'pending',
    page_count: readyHybrid || readyKeyword ? 12 : null,
    parsed_pages: readyHybrid || readyKeyword ? 12 : null,
    chunk_count: readyHybrid || readyKeyword ? 10 : null,
    keyword_status: readyHybrid || readyKeyword ? 'ready' : 'pending',
    embedding_status: readyHybrid ? 'ready' : (readyKeyword ? 'disabled' : 'pending'),
    embedded_count: readyHybrid ? 10 : 0,
    vector_coverage_pct: readyHybrid ? 100 : (readyKeyword ? 0 : null),
    vector_searchable: readyHybrid,
    embedding_model: readyHybrid ? 'BAAI/bge-m3' : null,
    embedding_dimension: readyHybrid ? 1024 : null,
    published_readiness: index.published_readiness,
    latest_job: index.job ? {
      job_id: index.job.job_id,
      job_type: 'index_pipeline',
      status: index.job.status,
      stage: index.job.stage,
      completed_work: index.job.completed_work,
      total_work: index.job.total_work,
      error_code: index.job.error_code,
      error_message: index.job.error_code,
      retry_count: index.job.retry_count,
      next_retry_at: index.job.next_retry_at,
      cancel_requested_at: index.job.cancel_requested_at,
    } : null,
    last_indexed_at: readyHybrid || readyKeyword ? date : null,
    warnings_count: 0,
    ...overrides,
  };
}

async function setup(initial?: DocumentDetails, opts?: { settings?: Partial<KnowledgeSettings>; siliconflowConfigured?: boolean }) {
  let document = initial;
  let listUnavailable = false;
  let deleteUnavailable = false;
  let indexUnavailable = false;
  let settings: KnowledgeSettings = { ...defaultSettings, ...opts?.settings };
  const sfConfigured = opts?.siliconflowConfigured ?? true;
  let index: Pick<DocumentIndex, 'job' | 'published_readiness'> = { job: null, published_readiness: null };
  const changeIndex = vi.fn<(_: string, __: RequestInit) => Response | Promise<Response>>();
  const indexAllCalls = vi.fn();
  const rebuildCalls = vi.fn();
  const settingsCalls = vi.fn();
  importer.mockResolvedValueOnce({ base_url: 'http://127.0.0.1:8766', ui_token: 'ui-test' });
  const fetchApi = vi.fn(async (url: string, options: RequestInit) => {
    expect(options.headers).toMatchObject({ Authorization: 'Bearer ui-test' });
    const path = new URL(url).pathname;
    if (options.method === 'GET' && path === '/api/v1/knowledge/index-status') {
      if (listUnavailable) return textbookUnavailable();
      const docs = document ? [toDocStatus(document, index)] : [];
      const activeCount = docs.filter((d) => ['queued', 'running', 'retry_wait'].includes(d.latest_job?.status ?? '')).length;
      const readyHybrid = docs.filter((d) => d.published_readiness === 'ready_hybrid').length;
      const readyKeyword = docs.filter((d) => d.published_readiness === 'ready_keyword').length;
      const overview: KnowledgeIndexOverview = {
        total_documents: docs.length,
        ready_hybrid: readyHybrid,
        ready_keyword: readyKeyword,
        in_progress: activeCount,
        pending: docs.length - readyHybrid - readyKeyword - activeCount,
        failed: 0,
        unsearchable: 0,
        missing_for_completion: docs.length - readyHybrid,
        active_job_id: activeCount > 0 ? (index.job?.job_id ?? null) : null,
        settings,
        siliconflow_configured: sfConfigured,
        documents: docs,
      };
      return Response.json(overview);
    }
    if (options.method === 'POST' && path === '/api/v1/knowledge/settings') {
      const patch = JSON.parse(String(options.body));
      settingsCalls(patch);
      settings = { ...settings, ...patch };
      return Response.json(settings);
    }
    if (options.method === 'POST' && path === '/api/v1/knowledge/index-all') {
      indexAllCalls();
      return Response.json({
        queued_count: document ? 1 : 0,
        skipped_complete: 0,
        skipped_active: 0,
        mode: settings.cloud_embedding_enabled ? 'hybrid' : 'local_only',
        queued_document_ids: document ? [document.document_id] : [],
      });
    }
    if (options.method === 'POST' && path === `${documentsPath}/${id}/rebuild-index`) {
      rebuildCalls(JSON.parse(String(options.body)));
      return Response.json({ document_id: id, queued: true, job_id: queued.job_id });
    }
    if (options.method === 'GET' && path === documentsPath) {
      if (listUnavailable) return textbookUnavailable();
      if (!document) return Response.json([]);
      const { versions, ...summary } = document;
      return Response.json([summary]);
    }
    if (options.method === 'GET' && path === `${documentsPath}/${id}`) return Response.json(document);
    if (options.method === 'GET' && path === `${documentsPath}/${id}/index`) {
      if (indexUnavailable) return textbookUnavailable();
      return Response.json({
        document_id: id, document_version_id: document?.versions.at(-1)?.document_version_id,
        ...index,
      });
    }
    if (options.method === 'POST' && path.startsWith(jobPath)) return changeIndex(path, options);
    if (options.method === 'POST' && path === `${documentsPath}/${id}/delete`) {
      if (deleteUnavailable) return textbookUnavailable();
      document = undefined;
      return new Response(null, { status: 204 });
    }
    throw new Error(`Unexpected request ${options.method} ${path}`);
  });
  vi.stubGlobal('fetch', fetchApi);
  const client = await connectApiClient();
  const view = render(<LibraryPanel client={client} active siliconflowConfigured={sfConfigured} />);
  return {
    fetchApi, changeIndex, indexAllCalls, rebuildCalls, settingsCalls,
    setDocument(value: DocumentDetails) { document = value; },
    setIndex(value: typeof index) { index = value; },
    makeIndexUnavailable(value: boolean) { indexUnavailable = value; },
    switchPanel(active: boolean) { view.rerender(<LibraryPanel client={client} active={active} siliconflowConfigured={sfConfigured} />); },
    makeListUnavailable() { listUnavailable = true; },
    makeDeleteUnavailable(value: boolean) { deleteUnavailable = value; },
  };
}

function button(name: string) {
  return screen.getByRole<HTMLButtonElement>('button', { name });
}

test('imports, shows real index status columns, and refreshes the list and open history after an update', async () => {
  const { setDocument } = await setup();
  await screen.findByText(/还没有教材/);
  const importPending = Promise.withResolvers<unknown>();
  importer.mockReturnValueOnce(importPending.promise);
  fireEvent.click(button('导入教材'));
  expect(button('导入教材').disabled).toBe(true);
  expect(screen.getByText('正在导入教材…')).toBeTruthy();
  expect(importer).toHaveBeenLastCalledWith('select_and_import_document', { request: { action: 'select' } });

  const receipt = { import_id: importId, document_id: id, document_version_id: original.versions[0].document_version_id };
  setDocument(original);
  importPending.resolve({ ...receipt, outcome: 'imported' });
  expect(await screen.findByText(/教材已导入/)).toBeTruthy();
  let table = within(await screen.findByRole('table', { name: '教材列表' }));
  expect(table.getByText(/9 B/)).toBeTruthy();
  expect(table.getByText('未解析')).toBeTruthy();
  await screen.findByRole('button', { name: '开始建立索引' });
  await screen.findByRole('heading', { name: '版本历史（1）' });

  importer.mockResolvedValueOnce({ ...receipt, outcome: 'duplicate' });
  fireEvent.click(button('导入教材'));
  expect(await screen.findByText(/此文件内容已在教材库中/)).toBeTruthy();
  expect(table.getAllByRole('row')).toHaveLength(2);

  const next = { ...original.versions[0], document_version_id: 'd'.repeat(32), size_bytes: 20 };
  setDocument({ ...original, version_count: 2, latest_size_bytes: next.size_bytes, versions: [...original.versions, next] });
  importer.mockResolvedValueOnce({ ...receipt, document_version_id: next.document_version_id, outcome: 'updated' });
  fireEvent.click(button('更新 教材.md'));
  expect(importer).toHaveBeenLastCalledWith('select_and_import_document', {
    request: { action: 'select', document_id: id },
  });
  expect(await screen.findByText(/教材已更新/)).toBeTruthy();
  await screen.findByRole('heading', { name: '版本历史（2）' });
  table = within(screen.getByRole('table', { name: '教材列表' }));
  expect(table.getByText(/20 B/)).toBeTruthy();
  expect(table.getByText(/v2/)).toBeTruthy();

  importer.mockResolvedValueOnce({ ...receipt, document_version_id: next.document_version_id, outcome: 'unchanged' });
  fireEvent.click(button('更新 教材.md'));
  expect(await screen.findByText('文件内容与最新版本相同，未新增版本。')).toBeTruthy();
});

test('supports one-time global cloud embedding consent, batch import, and one-click index-all', async () => {
  const { indexAllCalls, settingsCalls, setDocument } = await setup(original);
  await screen.findByRole('table', { name: '教材列表' });

  // One-time cloud embedding consent banner when enabling from OFF
  const cloudCheckbox = screen.getByLabelText(/允许云端向量化/) as HTMLInputElement;
  expect(cloudCheckbox.checked).toBe(false);
  fireEvent.click(cloudCheckbox);
  expect(screen.getByRole('group', { name: '确认开启全局云端向量化' })).toBeTruthy();
  expect(settingsCalls).not.toHaveBeenCalled();

  fireEvent.click(button('确认开启云端向量化'));
  await waitFor(() => expect(settingsCalls).toHaveBeenCalledWith({ cloud_embedding_enabled: true }));
  expect(await screen.findByText('已开启全局云端向量化授权。')).toBeTruthy();

  // One-click complete missing indexes
  fireEvent.click(button('一键补全索引（1）'));
  await waitFor(() => expect(indexAllCalls).toHaveBeenCalledTimes(1));
  expect(await screen.findByText(/已加入混合向量索引队列 1 本/)).toBeTruthy();

  // Batch multi-file import
  importer.mockResolvedValueOnce([
    { filename: 'a.pdf', ok: true, result: { import_id: importId, document_id: id, document_version_id: versionId, outcome: 'imported' } },
    { filename: 'b.pdf', ok: true, result: { import_id: importId, document_id: id, document_version_id: versionId, outcome: 'duplicate' } },
  ]);
  setDocument(original);
  fireEvent.click(button('批量导入教材'));
  expect(importer).toHaveBeenLastCalledWith('select_and_import_documents');
  expect(await screen.findByText(/批量导入完成：新增 1 本，重复跳过 1 本/)).toBeTruthy();
});

test('ignores a cancelled picker and resubmits only the Rust receipt ID after an uncertain result', async () => {
  const { fetchApi } = await setup();
  await screen.findByText(/还没有教材/);
  importer.mockResolvedValueOnce(null);
  await act(async () => fireEvent.click(button('导入教材')));
  expect(screen.queryByRole('alert')).toBeNull();
  expect(button('导入教材').disabled).toBe(false);
  expect(fetchApi).toHaveBeenCalledTimes(1);

  importer.mockRejectedValueOnce({ code: 'KNOWLEDGE_STORE_UNAVAILABLE', import_id: importId });
  fireEvent.click(button('导入教材'));
  expect((await screen.findByRole('alert')).textContent).toContain('教材知识库暂不可用');
  expect(button('导入教材').disabled).toBe(true);
  importer.mockRejectedValueOnce({ code: 'IMPORT_RETRY_NOT_FOUND' });
  fireEvent.click(button('重试本次导入'));
  expect(importer).toHaveBeenLastCalledWith('select_and_import_document', {
    request: { action: 'retry', import_id: importId },
  });
  await screen.findByText(/暂存记录已失效/);
  expect(button('导入教材').disabled).toBe(false);
});

test('preserves rows on list and delete failures, and requires confirmation before deleting', async () => {
  const { fetchApi, makeListUnavailable, makeDeleteUnavailable } = await setup(original);
  await screen.findByRole('rowheader', { name: /教材\.md/ });
  makeListUnavailable();
  fireEvent.click(button('刷新列表'));
  expect((await screen.findByRole('alert')).textContent).toContain('教材知识库暂不可用');
  expect(screen.getByRole('rowheader', { name: /教材\.md/ })).toBeTruthy();
  expect(button('刷新列表').disabled).toBe(false);

  fireEvent.click(button('删除 教材.md'));
  expect(fetchApi.mock.calls.some(([, options]) => options.method === 'POST')).toBe(false);
  fireEvent.click(button('取消删除'));
  expect(screen.queryByRole('button', { name: '确认删除' })).toBeNull();

  makeDeleteUnavailable(true);
  fireEvent.click(button('删除 教材.md'));
  fireEvent.click(button('确认删除'));
  await waitFor(() => expect(button('确认删除').disabled).toBe(false));
  expect(screen.getByRole('alert').textContent).toContain('教材知识库暂不可用');
  expect(screen.getByRole('rowheader', { name: /教材\.md/ })).toBeTruthy();

  makeDeleteUnavailable(false);
  fireEvent.click(button('确认删除'));
  expect(await screen.findByText('教材已删除。')).toBeTruthy();
  expect(screen.queryByRole('rowheader', { name: /教材\.md/ })).toBeNull();
});

test('discloses cloud transfer and waits for confirmation before submitting the latest version', async () => {
  const { changeIndex, setIndex } = await setup(original);
  await screen.findByRole('rowheader', { name: /教材\.md/ });
  fireEvent.click(button('查看 教材.md 详情'));
  fireEvent.click(await screen.findByRole('button', { name: '开始建立索引' }));
  expect(screen.getByText(/教材分块（Chunk）文本将发送到 SiliconFlow API/)).toBeTruthy();
  expect(changeIndex).not.toHaveBeenCalled();
  fireEvent.click(button('暂不提交'));
  expect(changeIndex).not.toHaveBeenCalled();

  changeIndex.mockImplementation((path, options) => {
    expect(path).toBe(jobPath);
    expect(JSON.parse(options.body as string)).toEqual({ allow_cloud_embedding: true });
    setIndex({ job: queued, published_readiness: null });
    return Response.json(queued);
  });
  fireEvent.click(button('开始建立索引'));
  fireEvent.click(button('同意并提交索引'));
  await screen.findByText('等待处理');
  expect(screen.getByRole('progressbar', { name: '当前阶段进度未确定' }).getAttribute('value')).toBeNull();
  expect(changeIndex).toHaveBeenCalledTimes(1);
});

test('keeps published keyword capability through real cancel acknowledgement and reuses the same version for retry', async () => {
  const { changeIndex, setIndex } = await setup(original);
  setIndex({ job: indexing, published_readiness: 'ready_keyword' });
  await screen.findByRole('rowheader', { name: /教材\.md/ });
  fireEvent.click(button('查看 教材.md 详情'));
  await screen.findByText('关键词检索可用');
  expect(screen.getByRole('progressbar', { name: '当前阶段进度' }).getAttribute('value')).toBe('1');
  expect(screen.getByText('当前阶段 1 / 3')).toBeTruthy();
  const mutation = Promise.withResolvers<Response>();
  changeIndex.mockImplementationOnce(() => mutation.promise);
  fireEvent.click(button('取消索引'));
  expect(screen.getByText('正在请求取消…')).toBeTruthy();
  expect(screen.queryByText('任务已取消')).toBeNull();
  const cancelling = { ...indexing, cancel_requested_at: 1795483800 };
  setIndex({ job: cancelling, published_readiness: 'ready_keyword' });
  mutation.resolve(Response.json(cancelling));
  await screen.findByText('正在取消');
  expect(button('取消索引').disabled).toBe(true);
  expect(screen.getByText('关键词检索可用')).toBeTruthy();

  setIndex({ job: { ...cancelling, status: 'cancelled' }, published_readiness: 'ready_keyword' });
  await screen.findByText('任务已取消', {}, { timeout: 2500 });
  changeIndex.mockImplementationOnce((path, options) => {
    expect(path).toBe(`${jobPath}/${queued.job_id}/retry`);
    expect(JSON.parse(options.body as string)).toEqual({ allow_cloud_embedding: true });
    const done = { ...queued, job_id: 'f'.repeat(32), status: 'succeeded' as const, stage: 'publishing' as const };
    setIndex({ job: done, published_readiness: 'ready_hybrid' });
    return Response.json(done);
  });
  fireEvent.click(button('重试索引'));
  fireEvent.click(button('同意并提交索引'));
  await screen.findByText('混合检索可用');
  expect(screen.getByText('处理完成')).toBeTruthy();
});

test('opens Retrieval Lab tab, runs debug-search without DeepSeek, and renders trace, candidates, rerank score, and warnings', async () => {
  const { fetchApi } = await setup(original, { settings: { cloud_embedding_enabled: true, cloud_rerank_enabled: true } });
  await screen.findByRole('rowheader', { name: /教材\.md/ });

  const originalFetch = fetchApi.getMockImplementation()!;
  fetchApi.mockImplementation(async (url: string, options: RequestInit) => {
    const path = new URL(url).pathname;
    if (options.method === 'POST' && path === '/api/v1/knowledge/debug-search') {
      const body = JSON.parse(String(options.body));
      expect(body.query).toBe('最大似然估计');
      expect(body.strategy).toBe('global');
      expect(body.rerank).toBe(true);
      return Response.json({
        query: '最大似然估计',
        retrieval_mode: 'hybrid',
        coverage_mode: 'global',
        actual_strategy: 'global',
        rerank_used: true,
        rerank_diagnostic: null,
        channel_status: 'keyword_only',
        hits: [
          {
            evidence_id: `ev1:${id}:1:chunk:${'1'.repeat(64)}:${'2'.repeat(64)}`,
            chunk_id: `chunk:${'1'.repeat(64)}`,
            chunk_set_id: 1,
            chunk_index: 0,
            document_id: id,
            document_version_id: versionId,
            textbook_name: '教材.md',
            format: 'pdf',
            section_path: ['第二章', '2.1 最大似然'],
            location_summary: '第 46 页（页标签 38）',
            location_precision: 'block',
            location_status: 'available',
            physical_pages: [46],
            page_labels: ['38'],
            line_range: null,
            paragraph_numbers: [],
            warnings: [{ code: 'PDF_UNMAPPED_GLYPHS', message: 'Unmapped math glyphs', physical_page: 46 }],
            historical: false,
            parse_spec_current: true,
            text_fidelity: 'parsed_text_unverified',
            body_text: '最大似然估计满足 $L(\\theta)$ 极值条件。',
            fusion_score: 0.0164,
            rerank_score: 0.912,
            channel_ranks: [{ channel: 'keyword', rank: 1, score: 4.25 }],
            keyword_matched_in: ['body'],
          },
        ],
        documents: [],
        diagnostic: { code: 'EMBEDDING_NOT_CONFIGURED', message: 'SiliconFlow Key 未配置' },
        trace: {
          trace_id: 'tr_test_1',
          original_query: '最大似然估计',
          resolved_query: '最大似然估计',
          query_variants: ['最大似然估计', 'maximum likelihood estimation'],
          requested_strategy: 'global',
          actual_strategy: 'global',
          requested_mode: 'hybrid',
          effective_channels: ['keyword'],
          in_scope_documents: 1,
          searchable_documents: 1,
          hit_documents: 1,
          cited_documents: 1,
          documents: [],
          stages: [
            { stage: 'bm25', status: 'ok', candidate_count: 1, elapsed_ms: 1.4 },
            { stage: 'rerank', status: 'ok', candidate_count: 1, elapsed_ms: 42.0 },
          ],
          candidates: [
            {
              chunk_id: `chunk:${'1'.repeat(64)}`,
              document_id: id,
              textbook_name: '教材.md',
              chunk_index: 0,
              page_range: 'p.46',
              section_path: ['第二章', '2.1 最大似然'],
              bm25_raw: 4.25,
              bm25_rank: 1,
              rrf_score: 0.0164,
              rrf_rank: 1,
              rerank_score: 0.912,
              rerank_rank: 1,
              selected: true,
            },
          ],
          final_evidence_ids: [`ev1:${id}:1:chunk:${'1'.repeat(64)}:${'2'.repeat(64)}`],
          prompt_evidence_count: 1,
          estimated_tokens: 40,
          degraded: true,
          degrade_reasons: ['EMBEDDING_NOT_CONFIGURED'],
          warnings: [],
          chinese_terms: ['最大似然估计'],
          english_terms: ['maximum likelihood estimation'],
          rewrite_mode: 'rules',
          total_elapsed_ms: 44.9,
        },
        timings: { scope_ms: 1.1, fts_ms: 1.4, vector_ms: 0, query_embedding_ms: 0, fusion_ms: 0.1, rerank_ms: 42.0, source_ms: 0.3 },
      });
    }
    return originalFetch(url, options);
  });

  fireEvent.click(screen.getByRole('tab', { name: /RAG 检索实验室/ }));
  fireEvent.change(screen.getByLabelText('输入检索实验室问题'), { target: { value: '最大似然估计' } });
  fireEvent.click(button('运行检索诊断'));
  expect(await screen.findByText('第 46 页（页标签 38）')).toBeTruthy();
  expect(screen.getByText('BGE-Reranker 已精排')).toBeTruthy();
  expect(screen.getByText('Rerank 0.912')).toBeTruthy();
  expect(screen.getByText(/章节：第二章 \/ 2\.1 最大似然/)).toBeTruthy();
  expect(screen.getByText(/PDF_UNMAPPED_GLYPHS/)).toBeTruthy();
  expect(screen.getByText(/最大似然估计满足/)).toBeTruthy();
  expect(screen.getAllByText(/maximum likelihood estimation/).length).toBeGreaterThanOrEqual(1);
});
