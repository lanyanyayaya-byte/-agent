import { invoke } from '@tauri-apps/api/core';
import { useEffect, useState } from 'react';
import type { ApiClient, FileGrant, GrantScope } from './api/apiClient';

type GrantList = { sessionId: string; items: FileGrant[] };
type Action = { scope: GrantScope } | { grant_id: string };

export default function GrantPanel({ client, sessionId }: { client: ApiClient; sessionId: string }) {
  const [list, setList] = useState<GrantList>();
  const [revision, setRevision] = useState(0);
  const [pending, setPending] = useState(false);
  const [loadError, setLoadError] = useState<string>();
  const [actionError, setActionError] = useState<{ sessionId: string; message: string }>();
  const items = list?.sessionId === sessionId ? list.items : undefined;

  useEffect(() => {
    const controller = new AbortController();
    setLoadError(undefined);
    void client.listGrants(sessionId, controller.signal).then((items) => {
      if (!controller.signal.aborted) setList({ sessionId, items });
    }).catch(() => {
      if (!controller.signal.aborted) {
        setList(undefined);
        setLoadError(sessionId);
      }
    });
    return () => controller.abort();
  }, [client, sessionId, revision]);

  async function changeGrant(action: Action) {
    setPending(true);
    setActionError(undefined);
    let refresh = true;
    try {
      if ('scope' in action) {
        const grant = await invoke<FileGrant | null>('select_file_grant', {
          request: { session_id: sessionId, scope: action.scope },
        });
        refresh = grant !== null;
      } else {
        await invoke<void>('revoke_file_grant', { request: { session_id: sessionId, grant_id: action.grant_id } });
        setList((previous) => previous?.sessionId === sessionId ? {
          sessionId, items: previous.items.filter((grant) => grant.grant_id !== action.grant_id),
        } : previous);
      }
    } catch (error) {
      const message = error === 'GRANT_PATH_REJECTED' ? '无法授权此资源。请选择可访问的 TXT、Markdown 文件或文件夹。'
        : error === 'GRANT_NOT_FOUND' ? '该授权已失效。'
          : 'scope' in action ? '无法完成授权，请重试。' : '无法撤销授权，请重试。';
      setActionError({ sessionId, message });
    } finally {
      if (refresh) setRevision((value) => value + 1);
      setPending(false);
    }
  }

  return (
    <section className="grants" aria-label="文件授权">
      <div className="grant-actions">
        <h2>文件授权</h2>
        <button type="button" disabled={pending} onClick={() => void changeGrant({ scope: 'file' })}>授权 TXT / MD</button>
        <button type="button" disabled={pending} onClick={() => void changeGrant({ scope: 'directory' })}>授权文件夹</button>
      </div>
      {loadError === sessionId ? <p role="alert">无法加载授权列表。 <button type="button" onClick={() => setRevision((value) => value + 1)}>重试加载</button></p>
        : items === undefined ? <p role="status">正在加载授权</p>
          : items.length === 0 && <p>此会话暂无文件授权。</p>}
      {actionError?.sessionId === sessionId && <p role="alert">{actionError.message}</p>}
      {items && items.length > 0 && (
        <ul>
          {items.map((grant) => (
            <li key={grant.grant_id}>
              <span>{grant.scope === 'file' ? '文件' : '文件夹'} · <bdi>{grant.display_name}</bdi></span>
              <button type="button" aria-label={`撤销 ${grant.display_name}`} disabled={pending}
                onClick={() => void changeGrant({ grant_id: grant.grant_id })}>撤销</button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
