import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import {
  ApiRequestError,
  isTerminal,
  type ApiClient,
  type CoverageMode,
  type DocumentSummary,
  type Message,
  type RetrievalMode,
  type RunState,
  type RunSummary,
  type SessionSummary,
  type StoredCitation,
  type ToolCallSummary,
} from './api/apiClient';
import type { RunEvent } from './api/chatEvents';
import GrantPanel from './GrantPanel';
import MarkdownAnswer from './MarkdownAnswer';

const MESSAGE_LIMIT = 20_000;
const POLL_INTERVAL_MS = 1000;
const SELECTED_SESSION_KEY = 'knowledgeagent.selected_session';
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
type ToolProgress = { id: string; name: string; state: 'running' | 'completed' | 'failed' | 'cancelled'; summary?: string };
type ToolEvent = Extract<RunEvent, { type: 'tool_started' | 'tool_completed' | 'tool_failed' }>;

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
  citationsByRun?: Record<string, StoredCitation[]>;
  live?: LiveTurn;
  submittingError?: string;
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

function failureMessage(code: string | null) {
  switch (code) {
    case 'MODEL_TIMEOUT': return 'DeepSeek 响应超时。';
    case 'MODEL_UPSTREAM_ERROR': return 'DeepSeek 调用失败。';
    case 'RUN_INTERRUPTED': return '后端运行已中断，本次生成已结束。';
    case 'RATE_LIMITED': return '本轮工具调用次数或返回内容已达到上限。';
    default: return '无法完成本次生成。';
  }
}

function toolProgress(previous: ToolProgress[], event: ToolEvent): ToolProgress[] {
  const id = event.data.tool_call_id;
  if (event.type === 'tool_started') {
    if (previous.some((tool) => tool.id === id)) return previous;
    return [...previous, { id, name: event.data.tool_name, state: 'running' }];
  }
  const summary = event.type === 'tool_completed' ? event.data.summary : event.data.message;
  const state = event.type === 'tool_completed' ? 'completed' : event.data.code === 'RUN_CANCELLED' ? 'cancelled' : 'failed';
  return previous.map((tool) => tool.id === id && tool.state === 'running' ? { ...tool, state, summary } : tool);
}

function ToolActivity({ tools, status }: { tools?: ToolProgress[]; status?: RunState }) {
  if (!tools?.length) return null;
  return (
    <ul className="tool-activity" aria-label="工具执行">
      {tools.map((tool) => (
        <li key={tool.id} role="status" data-state={tool.state}>
          <span className="tool-name">{tool.name}</span>{' · '}
          {tool.state === 'running' ? (status && isTerminal(status) ? '执行已中断' : '执行中')
            : { completed: '已完成', failed: '执行失败', cancelled: '已取消' }[tool.state]}
          {tool.summary && <span className="tool-summary"> · {tool.summary}</span>}
        </li>
      ))}
    </ul>
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
  }));
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
}: {
  client?: ApiClient;
  modelConfigured: boolean;
  preview?: boolean;
  defaultRagEnabled?: boolean;
  autoTitleSessions?: boolean;
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
  const [grantsOpen, setGrantsOpen] = useState(false);
  const [scopeOpen, setScopeOpen] = useState(false);
  const turnsEndRef = useRef<HTMLDivElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

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

  async function createSession() {
    if (!client || creating) return;
    const controller = new AbortController();
    requests.current.add(controller);
    setCreating(true);
    setCreateError('');
    try {
      const initialTitle = autoTitleSessions ? '新对话' : `会话 ${conversations.length + 1}`;
      const session = await client.createSession(initialTitle, controller.signal);
      if (controller.signal.aborted) return;
      setConversations((previous) => [{ ...session, input: '', version: 0 }, ...previous]);
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

  async function send(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!client || !active || !modelConfigured || busy || !loaded || tooLong) return;
    const { session_id: id } = active;
    const question = active.input.trim();
    if (!question) return;

    const requestId = crypto.randomUUID();
    const controller = new AbortController();
    requests.current.add(controller);
    const shouldAutoTitle = autoTitleSessions && DEFAULT_SESSION_TITLE.test(active.title);
    const nextAutoTitle = shouldAutoTitle ? deriveSessionTitle(question) : undefined;
    updateSession(id, (session) => ({
      ...session,
      title: nextAutoTitle && nextAutoTitle !== session.title ? nextAutoTitle : session.title,
      input: '',
      live: { requestId, question, draft: '' },
      submittingError: undefined,
    }));
    if (nextAutoTitle && nextAutoTitle !== active.title) {
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
            return { ...session, toolsByRun: { ...session.toolsByRun, [item.run_id]: items } };
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
          settle(id, item.run_id);
        }
      }, knowledgeOptions);
    } catch (error) {
      if (controller.signal.aborted) return;
      if (error instanceof ApiRequestError && !['DUPLICATE_CLIENT_REQUEST', 'RUN_ALREADY_ACTIVE'].includes(error.code ?? '')) {
        updateSession(id, (session) => ({
          ...session, live: undefined, input: session.input || question, submittingError: error.message,
        }));
        return;
      }
      updateSession(id, (session) => session.live?.requestId !== requestId ? session : {
        ...session, live: { ...session.live, draft: '', detached: true },
      });
      try {
        if (error instanceof ApiRequestError && error.code === 'RUN_ALREADY_ACTIVE') {
          updateSession(id, (session) => ({ ...session, live: undefined, submittingError: error.message }));
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
            ...session, submittingError: '暂时无法确认运行状态。',
            live: session.live && { ...session.live, lookupComplete: true },
          }));
        } else {
          updateSession(id, (session) => ({
            ...session, live: undefined, input: session.input || question,
            submittingError: '连接中断，暂时无法确认请求是否已提交。',
          }));
          reloadHistory(id);
        }
      }
    } finally {
      requests.current.delete(controller);
    }
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

      <div className="chat-stage">
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
                {ragEnabled && (
                  <button
                    type="button"
                    className={`topbar-chip ${scopeOpen ? 'active' : ''}`}
                    aria-expanded={scopeOpen}
                    onClick={() => {
                      setScopeOpen((prev) => !prev);
                      setGrantsOpen(false);
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
                  }}>
                  <span aria-hidden="true">📎</span>
                  <span>文件授权</span>
                </button>
              </div>
            </header>

            <div className={`context-drawer ${grantsOpen ? 'is-open' : 'is-closed'}`}>
              <GrantPanel client={client} sessionId={active.session_id} />
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
                          <ToolActivity tools={turnTools} status={status} />
                          {assistant && (
                            <MarkdownAnswer
                              ariaLabel="已保存回答"
                              messageId={assistant.message_id}
                              content={assistant.content}
                            />
                          )}
                          {status === 'failed' && run && (
                            <p role="alert" className="turn-error-banner">
                              {failureMessage(run.error_code)} {run.error_code && <code>{run.error_code}</code>}
                            </p>
                          )}
                          {user.run_id === live?.runId && showDraft && (
                            <MarkdownAnswer ariaLabel="回答草稿" content={live.draft} />
                          )}
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
                        {live.runId && <ToolActivity tools={active.toolsByRun?.[live.runId]} status={active.activeStatus} />}
                        {showDraft && <MarkdownAnswer ariaLabel="回答草稿" content={live.draft} />}
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
              <form ref={formRef} className="composer" onSubmit={send}>
                <label htmlFor="question" className="sr-only">问题</label>
                <textarea
                  id="question"
                  rows={2}
                  value={active.input}
                  placeholder="给 KnowledgeAgent 发送消息…（Enter 发送，Shift + Enter 换行）"
                  onKeyDown={handleComposerKeyDown}
                  onChange={(event) => {
                    const input = event.target.value;
                    updateSession(active.session_id, (session) => ({ ...session, input }));
                  }}
                />
                <div className="composer-bottom-bar">
                  <section className="knowledge-qa-inline" aria-label="教材知识库问答设置">
                    <label className={`rag-pill-toggle ${ragEnabled ? 'is-on' : ''}`}>
                      <input
                        type="checkbox"
                        aria-label="结合教材知识库"
                        checked={ragEnabled}
                        onChange={(e) => setRagEnabled(e.target.checked)}
                      />
                      <span>教材知识库 RAG</span>
                    </label>
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
                    {active.submittingError && <p role="alert" className="composer-hint">{active.submittingError}</p>}
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
