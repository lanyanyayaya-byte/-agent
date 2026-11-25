import { useEffect, useRef, useState } from 'react';
import {
  ApiRequestError, type ApiClient, type DocumentIndex, type IndexJobStage,
  type IndexJobState, type IndexJobStatus, type PublishedIndexReadiness,
} from './api/apiClient';

const jobLabels: Record<IndexJobStatus, string> = {
  queued: '等待处理',
  running: '正在处理',
  retry_wait: '等待重试',
  succeeded: '处理完成',
  failed: '任务失败',
  cancelled: '任务已取消',
};

const stageLabels: Record<IndexJobStage, string> = {
  parsing: '解析教材',
  chunking: '分块',
  keyword_indexing: '关键词索引',
  embedding: '向量索引（Embedding）',
  publishing: '确认检索能力',
};

const readinessLabels: Record<PublishedIndexReadiness, string> = {
  ready_keyword: '关键词检索可用',
  ready_hybrid: '混合检索可用',
  unsearchable: '无可检索正文',
};

const retryTime = new Intl.DateTimeFormat('zh-CN', { dateStyle: 'short', timeStyle: 'medium' });
const ACTIVE_POLL_MS = 1000;
const IDLE_POLL_MS = 5000;

function isActive(job: IndexJobState | null) {
  return job?.status === 'queued' || job?.status === 'running' || job?.status === 'retry_wait';
}

function jobError(code: string) {
  switch (code) {
    case 'EMBEDDING_NOT_CONFIGURED': return 'SiliconFlow Key 未配置。保存 Key 后可重试向量索引。';
    case 'EMBEDDING_AUTH_FAILED': return 'SiliconFlow 拒绝了 API Key（401）。请更新 Key 后重试。';
    case 'EMBEDDING_FORBIDDEN': return 'SiliconFlow 拒绝访问，请检查账号和模型权限。';
    case 'EMBEDDING_RATE_LIMITED': return 'SiliconFlow 请求达到限额（429）。';
    case 'EMBEDDING_NETWORK_ERROR': return '无法连接 SiliconFlow，请检查网络。';
    case 'EMBEDDING_TIMEOUT': return 'SiliconFlow 请求超时。';
    case 'EMBEDDING_SERVICE_ERROR': return 'SiliconFlow 服务暂时无法完成请求。';
    case 'INDEX_LEASE_EXPIRED': return '上次后台执行中断，已保存的索引产物会在恢复时复用。';
    case 'INDEX_PIPELINE_STORAGE_ERROR':
    case 'INDEX_PIPELINE_STORAGE_BUSY': return '教材知识库暂时无法完成读写。';
    default:
      if (/^(PDF|DOCX|MARKDOWN|DOCUMENT_PARSE)_/.test(code)) {
        return '无法解析教材，请检查文件是否可正常打开，并确认格式与内容有效。';
      }
      return '后台任务遇到错误，可查看诊断码。';
  }
}

function actionError(error: unknown) {
  switch (error instanceof ApiRequestError ? error.code : undefined) {
    case 'INDEX_VERSION_NOT_CURRENT': return '教材版本已变更，请确认最新版本后再操作。';
    case 'TEXTBOOK_NOT_FOUND':
    case 'INDEX_JOB_NOT_FOUND': return '教材或任务已不存在，请刷新状态。';
    case 'INDEX_JOB_NOT_RETRYABLE':
    case 'INDEX_JOB_NOT_TERMINAL':
    case 'INDEX_JOB_IDENTITY_MISMATCH': return '任务状态已变更，请确认最新状态后再操作。';
    case 'KNOWLEDGE_STORE_UNAVAILABLE': return '教材知识库暂不可用，请稍后再操作。';
    default: return '暂未确认操作结果，正在重新读取索引状态。';
  }
}

type Action = 'start' | 'retry' | 'cancel';
type Props = {
  client?: ApiClient;
  active: boolean;
  documentId: string;
  versionId: string;
  versionNumber: number;
  siliconflowConfigured: boolean;
  blocked: boolean;
  refreshLibrary: () => void;
};

export default function TextbookIndex({
  client, active, documentId, versionId, versionNumber, siliconflowConfigured, blocked, refreshLibrary,
}: Props) {
  const [snapshot, setSnapshot] = useState<{ source: ApiClient; index: DocumentIndex }>();
  const [readError, setReadError] = useState(false);
  const [error, setError] = useState('');
  const [pending, setPending] = useState<Action>();
  const [confirm, setConfirm] = useState<'start' | 'retry'>();
  const inFlight = useRef<AbortController>(undefined);
  const consentButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    return () => {
      inFlight.current?.abort();
      inFlight.current = undefined;
    };
  }, [active, client]);

  useEffect(() => {
    if (!active || !client) {
      setConfirm(undefined);
      setPending(undefined);
      setError('');
      return;
    }
    if (pending) return;
    const controller = new AbortController();
    let next: ReturnType<typeof setTimeout>;

    async function poll(source: ApiClient) {
      let delay = IDLE_POLL_MS;
      try {
        const state = await source.getDocumentIndex(documentId, controller.signal);
        if (controller.signal.aborted) return;
        if (state.document_version_id !== versionId) {
          refreshLibrary();
          return;
        }
        setSnapshot({ source, index: state });
        setReadError(false);
        if (isActive(state.job)) delay = ACTIVE_POLL_MS;
      } catch (error) {
        if (controller.signal.aborted) return;
        if (error instanceof ApiRequestError && error.code === 'TEXTBOOK_NOT_FOUND') {
          refreshLibrary();
          return;
        }
        setReadError(true);
      }
      next = setTimeout(() => void poll(source), delay);
    }

    void poll(client);
    return () => {
      controller.abort();
      clearTimeout(next);
    };
  }, [active, client, documentId, pending, refreshLibrary, versionId]);

  const index = snapshot?.source === client ? snapshot?.index : undefined;
  const job = index?.job ?? null;
  const disabled = !active || !client || !index || readError || !!pending || blocked;
  const cancelling = job?.status === 'running' && job.cancel_requested_at !== null;

  async function change(action: Action) {
    if (!client || disabled || inFlight.current || (action !== 'start' && !job)) return;
    const controller = new AbortController();
    inFlight.current = controller;
    setPending(action);
    setConfirm(undefined);
    setError('');
    try {
      let result: IndexJobState;
      if (action === 'start') {
        result = await client.submitDocumentIndex(documentId, versionId, controller.signal);
      } else if (job && action === 'retry') {
        result = await client.retryDocumentIndex(job, controller.signal);
      } else if (job) {
        result = await client.cancelDocumentIndex(job, controller.signal);
      } else {
        return;
      }
      if (!controller.signal.aborted) {
        setSnapshot((previous) => previous?.source === client
          ? { source: client, index: { ...previous.index, job: result } }
          : previous);
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setError(actionError(error));
        setReadError(true);
        if (error instanceof ApiRequestError && ['INDEX_VERSION_NOT_CURRENT', 'TEXTBOOK_NOT_FOUND'].includes(error.code ?? '')) {
          refreshLibrary();
        }
      }
    } finally {
      if (inFlight.current === controller) {
        inFlight.current = undefined;
        setPending(undefined);
      }
    }
  }

  function dismissConsent() {
    setConfirm(undefined);
    requestAnimationFrame(() => consentButton.current?.focus());
  }

  return (
    <section className="textbook-index" aria-label="教材索引">
      <h4>第 {versionNumber} 版索引（最新）</h4>
      {!index && client && !readError && <p role="status">正在读取索引状态…</p>}
      {(!client || readError) && (
        <p role="status">{index ? '暂时无法读取索引，下方为上次确认的状态。' : '暂时无法读取索引，请检查后端连接。'}</p>
      )}
      {index && (
        <dl className="index-status" aria-live="polite">
          <div><dt>索引任务</dt><dd>{cancelling ? '正在取消' : job ? jobLabels[job.status] : '已导入，尚未启动索引'}</dd></div>
          <div><dt>检索能力</dt><dd>{index.published_readiness ? readinessLabels[index.published_readiness] : '尚未发布索引'}</dd></div>
          {job && <div><dt>处理阶段</dt><dd>{stageLabels[job.stage]}</dd></div>}
          {job?.next_retry_at != null && (
            <div><dt>下次重试</dt><dd><time dateTime={new Date(job.next_retry_at * 1000).toISOString()}>
              {retryTime.format(job.next_retry_at * 1000)}
            </time></dd></div>
          )}
        </dl>
      )}
      {job && (
        <div className="index-progress">
          {job.total_work !== null && job.total_work > 0 ? (
            <><progress aria-label="当前阶段进度" value={job.completed_work} max={job.total_work} />
              <span>当前阶段 {job.completed_work} / {job.total_work}</span></>
          ) : isActive(job) && job.total_work === null ? (
            <><progress aria-label="当前阶段进度未确定" /><span>当前阶段工作量尚未确定</span></>
          ) : null}
        </div>
      )}
      {job?.error_code && <p className="index-error">{jobError(job.error_code)} <code>{job.error_code}</code></p>}
      {index?.published_readiness === 'unsearchable' && <p className="library-hint">当前文件没有可提取的正文。扫描 PDF 可能需要提供带文本层的版本。</p>}
      {client && !siliconflowConfigured && index
        && index.published_readiness !== 'ready_hybrid' && index.published_readiness !== 'unsearchable'
        && job?.error_code !== 'EMBEDDING_NOT_CONFIGURED' && (
        <p className="library-hint">SiliconFlow Key 未配置。仍可建立本地关键词索引，保存 Key 后可重试向量索引。</p>
      )}
      {error && <p role="alert">{error}</p>}
      {pending && <p role="status">{pending === 'cancel' ? '正在请求取消…' : '正在提交索引任务…'}</p>}
      <div className="library-actions">
        {index && !job && !['ready_hybrid', 'unsearchable'].includes(index.published_readiness ?? '') && (
          <button type="button" ref={consentButton} disabled={disabled || !!confirm} onClick={() => setConfirm('start')}>开始建立索引</button>
        )}
        {job && isActive(job) && (
          <button type="button" disabled={disabled || cancelling} onClick={() => void change('cancel')}>取消索引</button>
        )}
        {(job?.status === 'failed' || job?.status === 'cancelled') && (
          <button type="button" ref={consentButton} disabled={disabled || !!confirm} onClick={() => setConfirm('retry')}>重试索引</button>
        )}
      </div>
      {confirm && (
        <div className="index-consent" role="group" aria-label="确认云端向量索引">
          <p>为生成向量，教材分块（Chunk）文本将发送到 SiliconFlow API。原始文件不会整本上传。</p>
          <div className="library-actions">
            <button type="button" autoFocus disabled={disabled} onClick={() => void change(confirm)}>同意并提交索引</button>
            <button type="button" onClick={dismissConsent}>暂不提交</button>
          </div>
        </div>
      )}
    </section>
  );
}
