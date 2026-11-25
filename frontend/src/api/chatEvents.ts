type Envelope = { session_id: string; run_id: string };

export type ParseWarning = {
  code: string;
  message: string;
  physical_page?: number | null;
};

export type StoredCitation = {
  evidence_id: string;
  chunk_id: string;
  chunk_set_id: number;
  chunk_index: number;
  document_id: string;
  document_version_id: string;
  textbook_name: string;
  format: 'pdf' | 'docx' | 'md';
  section_path: string[];
  location_summary: string;
  location_precision: string;
  location_status: string;
  warnings: ParseWarning[];
  snippet: string;
  status: string;
  historical: boolean;
  parse_spec_current: boolean;
};

export type RunEvent = Envelope & (
  | { type: 'run_started'; data: { client_request_id: string } }
  | { type: 'message_delta'; data: { delta: string } }
  | { type: 'tool_started'; data: { tool_call_id: string; tool_name: string } }
  | { type: 'tool_completed'; data: { tool_call_id: string; summary: string } }
  | { type: 'tool_failed'; data: { tool_call_id: string; code: string; message: string } }
  | {
      type: 'run_citations';
      data: {
        citations: StoredCitation[];
        retrieval_query: string;
        coverage_mode: string;
        channel_status: string;
      };
    }
  | { type: 'run_completed'; data: { message_id: string } }
  | { type: 'error'; data: { code: string; message: string } }
  | { type: 'run_cancelled'; data: { reason: string } }
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function parseRunEvent(value: unknown): RunEvent | undefined {
  if (!isRecord(value) || value.version !== 1 || typeof value.ts !== 'string'
    || typeof value.session_id !== 'string' || typeof value.run_id !== 'string' || !isRecord(value.data)) {
    return;
  }

  const { session_id, run_id, type, data } = value;
  const envelope = { session_id, run_id };
  switch (type) {
    case 'run_started':
      if (typeof data.client_request_id === 'string') {
        return { ...envelope, type, data: { client_request_id: data.client_request_id } };
      }
      break;
    case 'message_delta':
      if (data.channel === 'text' && typeof data.delta === 'string') {
        return { ...envelope, type, data: { delta: data.delta } };
      }
      break;
    case 'tool_started':
      if (typeof data.tool_call_id === 'string' && data.tool_call_id && typeof data.tool_name === 'string') {
        return { ...envelope, type, data: { tool_call_id: data.tool_call_id, tool_name: data.tool_name } };
      }
      break;
    case 'tool_completed':
      if (typeof data.tool_call_id === 'string' && data.tool_call_id && typeof data.summary === 'string') {
        return { ...envelope, type, data: { tool_call_id: data.tool_call_id, summary: data.summary } };
      }
      break;
    case 'tool_failed':
      if (typeof data.tool_call_id === 'string' && data.tool_call_id
        && typeof data.code === 'string' && typeof data.message === 'string') {
        return { ...envelope, type, data: { tool_call_id: data.tool_call_id, code: data.code, message: data.message } };
      }
      break;
    case 'run_citations':
      if (Array.isArray(data.citations)) {
        return {
          ...envelope,
          type,
          data: {
            citations: data.citations as StoredCitation[],
            retrieval_query: typeof data.retrieval_query === 'string' ? data.retrieval_query : '',
            coverage_mode: typeof data.coverage_mode === 'string' ? data.coverage_mode : 'global',
            channel_status: typeof data.channel_status === 'string' ? data.channel_status : 'hybrid',
          },
        };
      }
      break;
    case 'run_completed':
      if (typeof data.message_id === 'string' && data.message_id) {
        return { ...envelope, type, data: { message_id: data.message_id } };
      }
      break;
    case 'error':
      if (typeof data.code === 'string' && typeof data.message === 'string') {
        return { ...envelope, type, data: { code: data.code, message: data.message } };
      }
      break;
    case 'run_cancelled':
      if (typeof data.reason === 'string') {
        return { ...envelope, type, data: { reason: data.reason } };
      }
      break;
    default:
      console.debug('Ignored an unsupported SSE event type.');
  }
}
