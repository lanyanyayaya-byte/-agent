export async function* readSse(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  let data: string[] = [];

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) return;
      pending += decoder.decode(value, { stream: true });
      let newLine = pending.indexOf('\n');
      while (newLine !== -1) {
        const line = pending.slice(0, newLine).replace(/\r$/, '');
        pending = pending.slice(newLine + 1);
        newLine = pending.indexOf('\n');

        if (line.startsWith('data:')) {
          data.push(line.slice(5).replace(/^ /, ''));
        }
        if (line === '' && data.length) {
          const frame = data.join('\n');
          data = [];
          let event: unknown;
          try {
            event = JSON.parse(frame);
          } catch {
            console.debug('Ignored malformed SSE data.');
            continue;
          }
          yield event;
        }
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
