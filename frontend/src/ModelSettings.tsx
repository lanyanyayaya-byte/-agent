import { invoke } from '@tauri-apps/api/core';
import { useRef, useState, type FormEvent } from 'react';

export type CredentialProvider = 'deepseek' | 'siliconflow';

const restoreErrors = new Map<string, string>([
  ['KEYCHAIN_ACCESS_FAILED', '无法从 macOS 钥匙串读取 SiliconFlow 凭据，请检查钥匙串权限或重新保存。'],
  ['KEYCHAIN_CREDENTIAL_CORRUPTED', 'macOS 钥匙串中的 SiliconFlow 凭据已损坏，请重新保存或清除。'],
  ['MODEL_CREDENTIAL_REPLAY_FAILED', '无法恢复 SiliconFlow 凭据，请重新保存或清除。'],
]);

const testErrors = new Map<string, string>([
  ['EMBEDDING_NOT_CONFIGURED', '请先保存 SiliconFlow API Key。'],
  ['EMBEDDING_AUTH_FAILED', 'SiliconFlow API Key 无效或已失效（401），请更新后重试。'],
  ['EMBEDDING_FORBIDDEN', 'SiliconFlow 拒绝访问（403），请检查账户和模型权限。'],
  ['EMBEDDING_RATE_LIMITED', 'SiliconFlow 请求达到限额（429），请稍后重试。'],
  ['EMBEDDING_TIMEOUT', '连接 SiliconFlow 超时，请稍后重试。'],
  ['EMBEDDING_NETWORK_ERROR', '无法连接 SiliconFlow，请检查网络后重试。'],
  ['EMBEDDING_SERVICE_ERROR', 'SiliconFlow 服务返回错误，请稍后重试。'],
  ['EMBEDDING_INVALID_RESPONSE', 'SiliconFlow 返回的模型或向量无效，预期为 BAAI/bge-m3 的 1024 维向量。'],
]);

export default function ModelSettings({
  provider,
  configured,
  restoreError,
  onConfiguredChange,
}: {
  provider: CredentialProvider;
  configured: boolean;
  restoreError?: string;
  onConfiguredChange: (provider: CredentialProvider, configured: boolean) => void;
}) {
  const siliconflow = provider === 'siliconflow';
  const name = siliconflow ? 'SiliconFlow' : 'DeepSeek';
  const field = useRef<HTMLInputElement>(null);
  const [pending, setPending] = useState<'save' | 'delete' | 'test'>();
  const [feedback, setFeedback] = useState<{ message: string; error?: boolean }>();
  const restoreMessage = !configured && restoreError ? restoreErrors.get(restoreError) : undefined;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const input = field.current;
    if (!input || pending) return;
    setFeedback(undefined);
    setPending('save');
    try {
      const submission = invoke<{ configured: boolean }>('save_provider_credential', {
        provider, apiKey: input.value.trim(),
      });
      input.value = '';
      if ((await submission).configured !== true) throw new Error();
      onConfiguredChange(provider, true);
      setFeedback({ message: siliconflow ? '已保存 SiliconFlow 凭据。' : '凭据已提交。' });
    } catch (error) {
      input.value = '';
      setFeedback({ error: true, message:
        error === 'KEYCHAIN_ACCESS_FAILED' || error === 'KEYCHAIN_CREDENTIAL_CORRUPTED'
          ? `无法访问 macOS Keychain，未能保存 ${name} 凭据。`
          : `未能提交 ${name} 凭据，请检查输入后重试。`,
      });
    } finally {
      setPending(undefined);
    }
  }

  async function remove() {
    if (pending) return;
    if (field.current) field.current.value = '';
    setFeedback(undefined);
    setPending('delete');
    try {
      const deletion = await invoke<{ configured: boolean }>('delete_provider_credential', {
        provider,
      });
      if (deletion.configured !== false) throw new Error();
      onConfiguredChange(provider, false);
      setFeedback({ message: `已删除 ${name} 凭据。` });
    } catch (error) {
      setFeedback({ error: true, message:
        error === 'KEYCHAIN_ACCESS_FAILED'
          ? `无法访问 macOS Keychain，未能删除 ${name} 凭据。`
          : `未能删除 ${name} 凭据，请重试。`,
      });
    } finally {
      setPending(undefined);
    }
  }

  async function testConnection() {
    if (pending || !configured) return;
    if (field.current) field.current.value = '';
    setFeedback(undefined);
    setPending('test');
    try {
      const result = await invoke<{ model: string; dimensions: number }>('test_siliconflow_connection');
      if (result.model !== 'BAAI/bge-m3' || result.dimensions !== 1024) throw 'EMBEDDING_INVALID_RESPONSE';
      setFeedback({ message: `连接成功，模型 ${result.model}，向量维度 ${result.dimensions}。` });
    } catch (error) {
      const message = typeof error === 'string' ? testErrors.get(error) : undefined;
      setFeedback({ error: true, message: message ?? '无法完成 SiliconFlow 连接测试，请稍后重试。' });
    } finally {
      setPending(undefined);
    }
  }

  return (
    <form className="model-settings" onSubmit={submit} autoComplete="off" aria-label={`${name} 凭据`}>
      <label htmlFor={`${provider}-key`}>{name} API Key</label>
      <div className="credential-input">
        <input
          id={`${provider}-key`} ref={field} type="password" required maxLength={512}
          autoComplete="off" spellCheck={false} autoCapitalize="none" disabled={pending !== undefined}
        />
        <button type="submit" disabled={pending !== undefined}>
          {siliconflow ? (configured ? '更新 Key' : '保存 Key') : (configured ? '更新' : '配置')}
        </button>
        {(configured || restoreMessage) && (
          <button type="button" disabled={pending !== undefined} onClick={() => void remove()}>
            {siliconflow ? '清除 Key' : '删除凭据'}
          </button>
        )}
        {siliconflow && (
          <button type="button" disabled={pending !== undefined || !configured} onClick={() => void testConnection()}
            title="使用已保存的 SiliconFlow API Key">
            {pending === 'test' ? '正在测试…' : '测试连接'}
          </button>
        )}
      </div>
      {siliconflow && <p role="status">SiliconFlow {configured ? '已配置' : '未配置'}</p>}
      {restoreMessage && <p role="alert">{restoreMessage}</p>}
      {feedback && <p role={feedback.error ? 'alert' : 'status'}>{feedback.message}</p>}
    </form>
  );
}
