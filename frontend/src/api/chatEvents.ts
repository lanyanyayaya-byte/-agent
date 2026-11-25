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

export type ToolSource = 'native' | 'mcp';
export type ToolCategory = 'skill' | 'workspace' | 'runtime';

export type ArtifactSummary = {
  artifact_id: string;
  session_id?: string;
  run_id?: string;
  filename: string;
  title: string | null;
  relative_path?: string;
  media_type: string;
  size_bytes: number;
  status: 'staged' | 'published' | 'discarded';
  created_at?: string;
  file_exists?: boolean;
};

export type RetrievalTraceSummary = {
  trace_id: string;
  actual_strategy: string;
  requested_mode: string;
  effective_channels: string[];
  in_scope_documents: number;
  searchable_documents: number;
  hit_documents: number;
  cited_documents: number;
  prompt_evidence_count: number;
  bm25_used: boolean;
  vector_used: boolean;
  rerank_used: boolean;
  total_elapsed_ms: number;
  context_expansion: number;
  degraded: boolean;
  degrade_reasons: string[];
  warnings: string[];
  error_code: string | null;
  error_message: string | null;
  summary_coverage?: Record<string, unknown> | null;
};

export type StageTraceView = {
  stage: string;
  status: 'ok' | 'skipped' | 'degraded' | 'failed';
  candidate_count: number;
  elapsed_ms: number;
  error_code?: string | null;
  message?: string | null;
};

export type CandidateTraceView = {
  chunk_id: string;
  document_id: string;
  textbook_name?: string | null;
  chunk_index: number;
  page_range?: string | null;
  section_path: string[];
  bm25_raw?: number | null;
  bm25_rank?: number | null;
  cosine?: number | null;
  vector_rank?: number | null;
  rrf_score?: number | null;
  rrf_rank?: number | null;
  rerank_score?: number | null;
  rerank_rank?: number | null;
  selected: boolean;
  exclusion_reason?: string | null;
  is_reference_chunk?: boolean;
  is_neighbor_expansion?: boolean;
  snippet?: string | null;
};

export type DocumentTraceView = {
  document_id: string;
  textbook_name?: string | null;
  version_id?: string | null;
  keyword_indexed: boolean;
  vector_indexed: boolean | null;
  searched_channels: string[];
  evidence_status: string;
  returned_count: number;
  skip_reason?: string | null;
};

export type RetrievalTraceData = {
  trace_id: string;
  original_query: string;
  resolved_query: string;
  query_variants: string[];
  requested_strategy: string;
  actual_strategy: string;
  requested_mode: string;
  effective_channels: string[];
  in_scope_documents: number;
  searchable_documents: number;
  hit_documents: number;
  cited_documents: number;
  documents: DocumentTraceView[];
  stages: StageTraceView[];
  candidates: CandidateTraceView[];
  final_evidence_ids: string[];
  prompt_evidence_count: number;
  estimated_tokens: number;
  degraded: boolean;
  degrade_reasons: string[];
  warnings: string[];
  chinese_terms?: string[];
  english_terms?: string[];
  rewrite_mode?: string;
  rewrite_reason?: string | null;
  bm25_used?: boolean;
  vector_used?: boolean;
  rerank_used?: boolean;
  total_elapsed_ms?: number;
  context_expansion?: number;
  summary_coverage?: {
    document_id?: string;
    textbook_name?: string;
    summary_mode?: string;
    total_chunks?: number;
    selected_chunks?: number;
    covered_sections?: string[];
    skipped_sections?: { section: string; reason: string }[];
    covered_pages_count?: number;
    uncovered_pages_count?: number;
    covered_page_range?: string;
  } | null;
  error_code?: string | null;
  error_message?: string | null;
  total_candidates_recorded?: number;
};

export type ToolEventMetadata = {
  source?: ToolSource;
  tool_category?: ToolCategory | null;
  recoverable?: boolean;
  server_id?: string | null;
  server_name?: string | null;
  original_tool_name?: string | null;
  duration_ms?: number | null;
};

export type RunEvent = Envelope & (
  | { type: 'run_started'; data: { client_request_id: string } }
  | { type: 'message_delta'; data: { delta: string } }
  | { type: 'tool_started'; data: { tool_call_id: string; tool_name: string } & Omit<ToolEventMetadata, 'duration_ms'> }
  | { type: 'tool_completed'; data: { tool_call_id: string; summary: string } & ToolEventMetadata }
  | { type: 'tool_failed'; data: { tool_call_id: string; code: string; message: string } & ToolEventMetadata }
  | {
      type: 'skill_activated';
      data: {
        skill_id: string;
        name: string;
        revision: string;
        dependencies_ready?: boolean;
        prepared_dependencies?: boolean;
      };
    }
  | {
      type: 'artifact_created';
      data: ArtifactSummary;
    }
  | {
      type: 'run_citations';
      data: {
        citations: StoredCitation[];
        retrieval_query: string;
        coverage_mode: string;
        channel_status: string;
      };
    }
  | {
      type: 'retrieval_trace';
      data: {
        summary: RetrievalTraceSummary;
        trace: RetrievalTraceData;
      };
    }
  | { type: 'run_completed'; data: { message_id: string; artifacts?: ArtifactSummary[] } }
  | { type: 'error'; data: { code: string; message: string } }
  | { type: 'run_cancelled'; data: { reason: string } }
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function parseToolMetadata(data: Record<string, unknown>, includeDuration: boolean): ToolEventMetadata {
  const meta: ToolEventMetadata = {};
  if (data.source === 'native' || data.source === 'mcp') meta.source = data.source;
  if (data.tool_category === 'skill' || data.tool_category === 'workspace' || data.tool_category === 'runtime'
    || data.tool_category === null) {
    meta.tool_category = data.tool_category;
  }
  if (typeof data.recoverable === 'boolean') meta.recoverable = data.recoverable;
  if (typeof data.server_id === 'string' || data.server_id === null) meta.server_id = data.server_id;
  if (typeof data.server_name === 'string' || data.server_name === null) meta.server_name = data.server_name;
  if (typeof data.original_tool_name === 'string' || data.original_tool_name === null) {
    meta.original_tool_name = data.original_tool_name;
  }
  if (includeDuration && (typeof data.duration_ms === 'number' || data.duration_ms === null)) {
    meta.duration_ms = data.duration_ms;
  }
  return meta;
}

function parseArtifactSummary(value: unknown): ArtifactSummary | undefined {
  if (!isRecord(value) || typeof value.artifact_id !== 'string' || typeof value.filename !== 'string') {
    return;
  }
  return {
    artifact_id: value.artifact_id,
    session_id: typeof value.session_id === 'string' ? value.session_id : undefined,
    run_id: typeof value.run_id === 'string' ? value.run_id : undefined,
    filename: value.filename,
    title: typeof value.title === 'string' ? value.title : null,
    relative_path: typeof value.relative_path === 'string' ? value.relative_path : undefined,
    media_type: typeof value.media_type === 'string' ? value.media_type : 'application/octet-stream',
    size_bytes: typeof value.size_bytes === 'number' ? value.size_bytes : 0,
    status: value.status === 'published' || value.status === 'discarded' ? value.status : 'staged',
    created_at: typeof value.created_at === 'string' ? value.created_at : undefined,
    file_exists: typeof value.file_exists === 'boolean' ? value.file_exists : undefined,
  };
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
        return {
          ...envelope,
          type,
          data: {
            tool_call_id: data.tool_call_id,
            tool_name: data.tool_name,
            ...parseToolMetadata(data, false),
          },
        };
      }
      break;
    case 'tool_completed':
      if (typeof data.tool_call_id === 'string' && data.tool_call_id && typeof data.summary === 'string') {
        return {
          ...envelope,
          type,
          data: {
            tool_call_id: data.tool_call_id,
            summary: data.summary,
            ...parseToolMetadata(data, true),
          },
        };
      }
      break;
    case 'tool_failed':
      if (typeof data.tool_call_id === 'string' && data.tool_call_id
        && typeof data.code === 'string' && typeof data.message === 'string') {
        return {
          ...envelope,
          type,
          data: {
            tool_call_id: data.tool_call_id,
            code: data.code,
            message: data.message,
            ...parseToolMetadata(data, true),
          },
        };
      }
      break;
    case 'skill_activated':
      if (typeof data.skill_id === 'string' && typeof data.name === 'string') {
        return {
          ...envelope,
          type,
          data: {
            skill_id: data.skill_id,
            name: data.name,
            revision: typeof data.revision === 'string' ? data.revision : '',
            dependencies_ready: typeof data.dependencies_ready === 'boolean' ? data.dependencies_ready : undefined,
            prepared_dependencies: typeof data.prepared_dependencies === 'boolean' ? data.prepared_dependencies : undefined,
          },
        };
      }
      break;
    case 'artifact_created': {
      const artifact = parseArtifactSummary(data);
      if (artifact) {
        return { ...envelope, type, data: artifact };
      }
      break;
    }
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
    case 'retrieval_trace':
      if (isRecord(data.summary) && isRecord(data.trace)) {
        return {
          ...envelope,
          type,
          data: {
            summary: data.summary as RetrievalTraceSummary,
            trace: data.trace as RetrievalTraceData,
          },
        };
      }
      break;
    case 'run_completed':
      if (typeof data.message_id === 'string' && data.message_id) {
        const artifacts = Array.isArray(data.artifacts)
          ? data.artifacts.map(parseArtifactSummary).filter((a): a is ArtifactSummary => Boolean(a))
          : undefined;
        return {
          ...envelope,
          type,
          data: {
            message_id: data.message_id,
            ...(artifacts && artifacts.length > 0 ? { artifacts } : {}),
          },
        };
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
