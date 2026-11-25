import { invoke, isTauri } from '@tauri-apps/api/core';
import { useEffect, useRef, useState } from 'react';
import { connectApiClient, type ApiClient, type BackendStatus } from './api/apiClient';
import Chat from './Chat';
import LibraryPanel from './LibraryPanel';
import ModelSettings, { type CredentialProvider } from './ModelSettings';
import './styles.css';

type ConnectionState =
  | { phase: 'starting_sidecar' | 'connecting_backend' }
  | { phase: 'ready'; status: BackendStatus; client: ApiClient; siliconflowRestoreError?: string }
  | { phase: 'connection_error'; message: string };

const labels = {
  starting_sidecar: '正在获取桌面连接配置',
  connecting_backend: '正在连接后端',
  ready: '后端已连接',
  connection_error: '后端连接失败',
};

export default function App() {
  if (!isTauri()) {
    return (
      <main className="preview-shell">
        <header className="preview-header">
          <h1>KnowledgeAgent</h1>
        </header>
        <section className="connection preview-card" aria-labelledby="preview-heading">
          <h2 id="preview-heading">网页预览</h2>
          <p>请切换到 KnowledgeAgent 桌面窗口使用聊天、文件授权和教材导入。</p>
        </section>
        <div className="preview-chat-wrap">
          <Chat modelConfigured={false} preview />
        </div>
      </main>
    );
  }
  return (
    <main className="app-shell">
      <DesktopWorkspace />
    </main>
  );
}

function DesktopWorkspace() {
  const [panel, setPanel] = useState<'chat' | 'library'>('chat');
  const [connection, setConnection] = useState<ConnectionState>({ phase: 'starting_sidecar' });
  const [attempt, setAttempt] = useState(0);
  const credentialRevision = useRef(0);
  const [logPending, setLogPending] = useState(false);
  const [logError, setLogError] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const autoOpenedSettings = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    let nextCheck: ReturnType<typeof setTimeout> | undefined;

    function showError(error: unknown) {
      if (!controller.signal.aborted) {
        setConnection({
          phase: 'connection_error',
          message: error instanceof Error ? error.message : '无法连接后端。',
        });
      }
    }

    async function connect() {
      const client = await connectApiClient();
      if (controller.signal.aborted) return;
      setConnection({ phase: 'connecting_backend' });

      async function checkStatus() {
        const revision = credentialRevision.current;
        const status = await client.getStatus(controller.signal);
        if (controller.signal.aborted) return;
        const current = revision === credentialRevision.current;
        setConnection((previous) => {
          const sameConnection = previous.phase === 'ready' && previous.client === client;
          if (sameConnection && !current) return previous;
          return {
            phase: 'ready', status, client,
            siliconflowRestoreError: sameConnection ? previous.siliconflowRestoreError : client.siliconflowRestoreError,
          };
        });
        nextCheck = setTimeout(() => void checkStatus().catch(showError), 3000);
      }

      await checkStatus();
    }

    void connect().catch(showError);
    return () => {
      controller.abort();
      clearTimeout(nextCheck);
    };
  }, [attempt]);

  useEffect(() => {
    if (connection.phase === 'ready' && !autoOpenedSettings.current) {
      if (!connection.status.model_configured || connection.siliconflowRestoreError) {
        setSettingsOpen(true);
      }
      autoOpenedSettings.current = true;
    }
  }, [connection]);

  function retry() {
    setConnection({ phase: 'starting_sidecar' });
    setAttempt((previous) => previous + 1);
  }

  async function openLogDirectory() {
    if (logPending) return;
    setLogError('');
    setLogPending(true);
    try {
      await invoke('open_log_directory');
    } catch {
      setLogError('无法打开日志目录，请检查应用数据目录权限。');
    } finally {
      setLogPending(false);
    }
  }

  const client = connection.phase === 'ready' ? connection.client : undefined;
  const needsSetup = connection.phase === 'ready' && (!connection.status.model_configured || !!connection.siliconflowRestoreError);

  function changeCredentialStatus(provider: CredentialProvider, configured: boolean) {
    credentialRevision.current += 1;
    setConnection((previous) => {
      if (previous.phase !== 'ready' || previous.client !== client) return previous;
      if (provider === 'deepseek') {
        return { ...previous, status: { ...previous.status, model_configured: configured } };
      }
      return {
        ...previous,
        siliconflowRestoreError: undefined,
        status: { ...previous.status, siliconflow_configured: configured },
      };
    });
  }

  return (
    <div className="desktop-layout">
      <aside className="app-sidebar">
        <div className="sidebar-brand">
          <div className="brand-logo" aria-hidden="true">KA</div>
          <div className="brand-copy">
            <h1>KnowledgeAgent</h1>
            <span className="brand-subtitle">RAG 教材学术助教</span>
          </div>
        </div>

        <nav className="workspace-nav" aria-label="工作区">
          <button
            type="button"
            className="nav-item"
            aria-controls="chat-panel"
            aria-pressed={panel === 'chat'}
            onClick={() => setPanel('chat')}>
            <span className="nav-icon" aria-hidden="true">💬</span>
            <span>聊天</span>
          </button>
          <button
            type="button"
            className="nav-item"
            aria-controls="library-panel"
            aria-pressed={panel === 'library'}
            onClick={() => setPanel('library')}>
            <span className="nav-icon" aria-hidden="true">📚</span>
            <span>教材知识库</span>
          </button>
        </nav>

        <div className="sidebar-spacer" />

        <section className="connection sidebar-footer" aria-label="后端连接">
          <div className="status-bar">
            <div role="status" className="status-block">
              <div className="status-heading">
                <span className="indicator" data-phase={connection.phase} aria-hidden="true" />
                <h2>{labels[connection.phase]}</h2>
              </div>
              <code>{connection.phase}</code>
            </div>
          </div>

          {connection.phase === 'ready' && (
            <dl className="system-metrics">
              <div>
                <dt>数据库</dt>
                <dd>{connection.status.database === 'ready' ? '已就绪' : '未配置'}</dd>
              </div>
              <div hidden={panel !== 'chat'}>
                <dt>模型</dt>
                <dd>{connection.status.model_configured ? '已配置' : '未配置'}</dd>
              </div>
            </dl>
          )}

          {connection.phase === 'connection_error' && (
            <div className="connection-error-box">
              <p role="alert">{connection.message}</p>
              <button type="button" className="sidebar-action-btn primary" onClick={retry}>重试</button>
            </div>
          )}

          <div className="sidebar-actions">
            {connection.phase === 'ready' && (
              <button
                type="button"
                className={`sidebar-action-btn ${settingsOpen ? 'active' : ''} ${needsSetup ? 'needs-attention' : ''}`}
                aria-expanded={settingsOpen}
                onClick={() => setSettingsOpen((prev) => !prev)}>
                <span aria-hidden="true">⚙️</span>
                <span>模型与凭据设置</span>
              </button>
            )}
            <button
              type="button"
              className="sidebar-action-btn"
              disabled={logPending}
              onClick={() => void openLogDirectory()}>
              <span aria-hidden="true">📂</span>
              <span>打开日志目录</span>
            </button>
          </div>
          {logError && <p role="alert" className="sidebar-alert">{logError}</p>}

          {connection.phase === 'ready' && (
            <>
              {settingsOpen && (
                <div
                  className="modal-backdrop"
                  aria-hidden="true"
                  onClick={() => setSettingsOpen(false)}
                />
              )}
              <div className={`credentials-popover ${settingsOpen ? 'is-open' : 'is-closed'}`}>
                <div className="popover-header">
                  <div>
                    <h3>模型与向量凭据设置</h3>
                    <p>所有 API Key 均通过 macOS 钥匙串本地加密存储</p>
                  </div>
                  <button
                    type="button"
                    className="icon-close-btn"
                    aria-label="关闭凭据设置"
                    onClick={() => setSettingsOpen(false)}>
                    ✕
                  </button>
                </div>
                <div className="credentials-grid">
                  {panel === 'chat' && (
                    <ModelSettings
                      provider="deepseek"
                      configured={connection.status.model_configured}
                      onConfiguredChange={changeCredentialStatus}
                    />
                  )}
                  <ModelSettings
                    provider="siliconflow"
                    configured={connection.status.siliconflow_configured === true}
                    restoreError={connection.siliconflowRestoreError}
                    onConfiguredChange={changeCredentialStatus}
                  />
                </div>
              </div>
            </>
          )}
        </section>
      </aside>

      <div className="workspace-main">
        <div id="chat-panel" className="workspace-panel chat-workspace-panel" hidden={panel !== 'chat'}>
          <Chat
            client={client}
            modelConfigured={connection.phase === 'ready' && connection.status.model_configured}
            defaultRagEnabled
            autoTitleSessions
          />
        </div>
        <div id="library-panel" className="workspace-panel library-workspace-panel" hidden={panel !== 'library'}>
          <LibraryPanel
            client={client}
            active={panel === 'library'}
            siliconflowConfigured={connection.phase === 'ready' && connection.status.siliconflow_configured === true}
          />
        </div>
      </div>
    </div>
  );
}
