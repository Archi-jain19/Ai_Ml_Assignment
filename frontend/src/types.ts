export type ResultStatus = 'scored' | 'insufficient_evidence' | 'not_observable' | 'unsuitable';

export interface FacetResult {
  facet: string;
  status: ResultStatus;
  score?: number | null;
  score_label?: string | null;
  confidence: number;
  reason: string;
  evidence?: string | null;
  facet_type: string;
  scoring_definition: string;
  abstention_policy: string;
}

export interface RetrievedFacetMeta {
  facet: string;
  similarity: number;
  facet_type: string;
  definition: string;
}

export interface AnalysisMetrics {
  num_facets_retrieved: number;
  num_scored: number;
  num_insufficient_evidence: number;
  num_not_observable: number;
  average_confidence?: number;
  processing_time_ms?: number;
  coverage_label: string;
  coverage_percentage: number;
}

export interface EvaluateResponse {
  conversation_id: string;
  conversation: string;
  metrics: AnalysisMetrics;
  results: FacetResult[];
  retrieved_facets: RetrievedFacetMeta[];
}

export interface TaxonomyItem {
  facet: string;
  raw_facet: string;
  facet_type: string;
  conversation_observable: boolean;
  sensitivity: string;
  scoring_definition: string;
  score_1_anchor?: string;
  score_2_anchor?: string;
  score_3_anchor?: string;
  score_4_anchor?: string;
  score_5_anchor?: string;
  abstention_policy: string;
}

export interface BenchmarkSummary {
  total_benchmark_examples: number;
  total_reference_cases: number;
  evaluated_cases: number;
  agreement_percentage: number;
  correct_abstentions_percentage: number;
  incorrect_scores_percentage: number;
  incorrect_abstentions_percentage: number;
  exact_score_accuracy: number;
  score_mae: number;
  recall_at_10?: number;
  recall_at_20?: number;
  mrr?: number;
}

export interface BenchmarkComparisonRow {
  conversation_id: string;
  conversation_type: string;
  conversation_text: string;
  facet: string;
  expected_status: ResultStatus;
  expected_score?: number | null;
  predicted_status: ResultStatus;
  predicted_score?: number | null;
  predicted_reason: string;
  human_reasoning: string;
  result_badge: 'agreement' | 'partial' | 'error';
}

export interface BenchmarkResponse {
  summary: BenchmarkSummary;
  conversations: Array<{
    conversation_id: string;
    type: string;
    text: string;
  }>;
  comparison_rows: BenchmarkComparisonRow[];
}
