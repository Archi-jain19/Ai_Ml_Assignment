import React from 'react';
import { ArrowDown, Database, Cpu, Layers, ShieldCheck, Zap, FileCode, CheckCircle } from 'lucide-react';

export const SystemOverview: React.FC = () => {
  const steps = [
    { title: 'Conversation Input', desc: 'Raw text snippet ingested and normalized.', icon: FileCode },
    { title: 'Preprocessing', desc: 'Attribution parsing, code-switch detection, negative context extraction.', icon: Layers },
    { title: 'Facet Taxonomy & Rules', desc: '399+ enriched facets filtered by observability rules.', icon: Database },
    { title: 'Embedding Retrieval (FAISS)', desc: 'all-MiniLM-L6-v2 vector indexing prunes search space to top-K.', icon: Zap },
    { title: 'Relevant Facet Subset', desc: '20–50 candidate facets routed for scoring.', icon: Layers },
    { title: 'Batch Scoring Engine', desc: 'Llama 3.3 70B via Groq + deterministic grounded rules.', icon: Cpu },
    { title: 'Validation & Bounds Parsing', desc: 'Strict JSON schema bounds check and non-hallucination validation.', icon: CheckCircle },
    { title: 'Score OR Abstain', desc: 'Score (1–5) if explicit evidence exists; Abstain if ambiguous.', icon: ShieldCheck },
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-8 py-6">
      {/* Top Title Banner */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 sm:p-8 shadow-xs text-center max-w-3xl mx-auto">
        <span className="text-xs font-semibold px-3 py-1 rounded-full bg-slate-100 text-slate-700 border border-slate-200 uppercase tracking-wider mb-2 inline-block">
          Architecture Blueprint
        </span>
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight mt-1">
          Evidence-Grounded Facet Evaluation Architecture
        </h1>
        <p className="text-sm text-slate-600 mt-2 leading-relaxed">
          How FacetLens scales retrieval and scoring across thousands of behavioral facets while strictly enforcing zero-hallucination abstention guardrails.
        </p>
      </div>

      {/* Designed for Scale Callout */}
      <div className="bg-slate-900 text-white rounded-2xl p-8 shadow-lg relative overflow-hidden">
        <div className="relative z-10 max-w-3xl">
          <span className="text-xs font-bold text-emerald-400 uppercase tracking-widest block mb-1">
            Engineered for Scale
          </span>
          <h2 className="text-3xl font-extrabold tracking-tight text-white mb-3">
            5,000+ Facet Taxonomy Library
          </h2>
          <p className="text-slate-300 text-sm leading-relaxed mb-6">
            Sending thousands of facet definitions into an LLM prompt context window is slow, expensive, and causes severe attention degradation. FacetLens retrieves a targeted candidate subset before scoring.
          </p>

          <div className="inline-flex items-center space-x-3 bg-slate-800/80 border border-slate-700 px-4 py-3 rounded-xl text-xs font-mono text-slate-200">
            <span className="font-bold text-emerald-400">5,000 Facets</span>
            <span className="text-slate-500">→</span>
            <span className="font-bold text-blue-400">FAISS Index</span>
            <span className="text-slate-500">→</span>
            <span className="font-bold text-purple-400">Top 20 Candidates</span>
            <span className="text-slate-500">→</span>
            <span className="font-bold text-emerald-400">Score OR Abstain</span>
          </div>
        </div>
      </div>

      {/* Visual Pipeline Flowchart */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 sm:p-8 shadow-xs">
        <h3 className="text-lg font-bold text-slate-900 mb-6 text-center">Execution Pipeline Flowchart</h3>

        <div className="max-w-2xl mx-auto space-y-3">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <React.Fragment key={idx}>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex items-center justify-between shadow-2xs hover:border-slate-300 transition-colors">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 rounded-lg bg-slate-900 text-white flex items-center justify-center text-xs font-bold shrink-0">
                      {idx + 1}
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-slate-900">{step.title}</h4>
                      <p className="text-xs text-slate-500">{step.desc}</p>
                    </div>
                  </div>
                  <Icon className="w-5 h-5 text-slate-400 shrink-0 ml-2" />
                </div>

                {idx < steps.length - 1 && (
                  <div className="flex justify-center my-1">
                    <ArrowDown className="w-4 h-4 text-slate-300" />
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* 5 Feature Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
          <div className="w-8 h-8 rounded-lg bg-blue-50 text-blue-700 flex items-center justify-center mb-3">
            <Zap className="w-4 h-4" />
          </div>
          <h4 className="text-sm font-bold text-slate-900 mb-1">Embedding Retrieval</h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            SentenceTransformers + FAISS vector search prunes thousands of candidate facets down to the top 20–50 most relevant items.
          </p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
          <div className="w-8 h-8 rounded-lg bg-purple-50 text-purple-700 flex items-center justify-center mb-3">
            <Cpu className="w-4 h-4" />
          </div>
          <h4 className="text-sm font-bold text-slate-900 mb-1">Batch Scoring & Routing</h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            Routes selected facets to Groq Llama 3.3 70B in optimized batches, evaluating evidence across all candidate facets in parallel.
          </p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
          <div className="w-8 h-8 rounded-lg bg-emerald-50 text-emerald-700 flex items-center justify-center mb-3">
            <Database className="w-4 h-4" />
          </div>
          <h4 className="text-sm font-bold text-slate-900 mb-1">Pre-Computed Metadata Caching</h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            Pre-computes and caches 384-dimensional embeddings and taxonomy definitions for fast sub-millisecond retrieval.
          </p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
          <div className="w-8 h-8 rounded-lg bg-amber-50 text-amber-700 flex items-center justify-center mb-3">
            <CheckCircle className="w-4 h-4" />
          </div>
          <h4 className="text-sm font-bold text-slate-900 mb-1">Structured Validation</h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            Validates output schema types, score ranges (1–5), confidence bounds (0.0–1.0), and enforces exact reason attribution.
          </p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs md:col-span-2">
          <div className="w-8 h-8 rounded-lg bg-red-50 text-red-700 flex items-center justify-center mb-3">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <h4 className="text-sm font-bold text-slate-900 mb-1">Zero-Hallucination Abstention Policy</h4>
          <p className="text-xs text-slate-600 leading-relaxed">
            If the conversation lacks concrete proof for a facet (or if the facet measures unobservable medical/biographical data), the system explicitly returns <span className="font-semibold text-amber-800">INSUFFICIENT_EVIDENCE</span> or <span className="font-semibold text-slate-800">NOT_OBSERVABLE</span> rather than hallucinating a score.
          </p>
        </div>
      </div>
    </div>
  );
};
