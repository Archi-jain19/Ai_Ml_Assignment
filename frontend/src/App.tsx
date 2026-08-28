import React, { useEffect, useState, useMemo } from 'react';
import type { EvaluateResponse } from './types';
import { Header } from './components/Header';
import type { PageTab } from './components/Header';
import { ConversationInput } from './components/ConversationInput';
import { AnalysisSummary } from './components/AnalysisSummary';
import { FacetFilterBar } from './components/FacetFilterBar';
import type { FilterStatus } from './components/FacetFilterBar';
import { FacetResultsList } from './components/FacetResultsList';
import { FacetDetailModal } from './components/FacetDetailModal';
import { SystemOverview } from './components/SystemOverview';
import { BenchmarkPage } from './components/BenchmarkPage';
import { EmptyState } from './components/EmptyState';
import { ErrorState } from './components/ErrorState';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<PageTab>('analyzer');
  const [modelReady, setModelReady] = useState<boolean>(true);

  // Input & Evaluation State
  const [conversationText, setConversationText] = useState<string>('');
  const [evalOutput, setEvalOutput] = useState<EvaluateResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [activeStage, setActiveStage] = useState<'retrieval' | 'scoring' | 'validation' | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Filter & Search State for Page 2 Results
  const [activeFilter, setActiveFilter] = useState<FilterStatus>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Selected Facet Detail Modal State
  const [selectedFacetName, setSelectedFacetName] = useState<string | null>(null);

  const defaultSnippet =
    'I struggled to manage several projects at once. After missing a few internal deadlines, I reviewed my workflow, created a weekly task plan, and started prioritizing important tasks first. I have followed this system consistently for the past three months and now complete my projects before their deadlines.';

  // Check Backend Status on Mount
  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => {
        if (data.model_ready) setModelReady(true);
      })
      .catch(() => setModelReady(false));
  }, []);

  // Handle Conversation Analysis
  const handleAnalyze = async (overrideText?: string) => {
    const textToAnalyze = (overrideText || conversationText).trim();
    if (!textToAnalyze) return;

    setIsLoading(true);
    setError(null);

    // Stage 1: Retrieval
    setActiveStage('retrieval');

    try {
      // Stage 2: Scoring transition timer
      setTimeout(() => setActiveStage('scoring'), 400);

      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation: textToAnalyze, top_k: 20 }),
      });

      // Stage 3: Validation transition
      setActiveStage('validation');

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || 'The scoring service returned an invalid or incomplete response.');
      }

      const data: EvaluateResponse = await res.json();
      setEvalOutput(data);
    } catch (err: any) {
      setError(err.message || 'Analysis failed. Please verify the scoring server.');
    } finally {
      setIsLoading(false);
      setActiveStage(null);
    }
  };

  const handleClear = () => {
    setConversationText('');
    setEvalOutput(null);
    setError(null);
    setActiveFilter('all');
    setSearchQuery('');
  };

  const handleLoadExample = () => {
    setConversationText(defaultSnippet);
    handleAnalyze(defaultSnippet);
  };

  const handleSelectSampleText = (text: string) => {
    setConversationText(text);
    setActiveTab('analyzer');
    handleAnalyze(text);
  };

  // Compute Filter Counts & Filtered Results List
  const resultsList = evalOutput?.results || [];

  const counts = useMemo(() => {
    return {
      all: resultsList.length,
      scored: resultsList.filter((r) => r.status === 'scored').length,
      insufficient_evidence: resultsList.filter((r) => r.status === 'insufficient_evidence').length,
      not_observable: resultsList.filter((r) => r.status === 'not_observable' || r.status === 'unsuitable').length,
    };
  }, [resultsList]);

  const filteredResults = useMemo(() => {
    return resultsList.filter((r) => {
      // Status Filter
      if (activeFilter === 'scored' && r.status !== 'scored') return false;
      if (activeFilter === 'insufficient_evidence' && r.status !== 'insufficient_evidence') return false;
      if (activeFilter === 'not_observable' && r.status !== 'not_observable' && r.status !== 'unsuitable') return false;

      // Search Filter
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesFacet = r.facet.toLowerCase().includes(q);
        const matchesReason = r.reason.toLowerCase().includes(q);
        const matchesEvidence = r.evidence ? r.evidence.toLowerCase().includes(q) : false;
        return matchesFacet || matchesReason || matchesEvidence;
      }

      return true;
    });
  }, [resultsList, activeFilter, searchQuery]);

  const currentResultObject = useMemo(() => {
    if (!selectedFacetName || !evalOutput) return null;
    return evalOutput.results.find((r) => r.facet.toLowerCase() === selectedFacetName.toLowerCase()) || null;
  }, [selectedFacetName, evalOutput]);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* Header */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} modelReady={modelReady} />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'analyzer' && (
          <div className="space-y-6">
            {/* PAGE 1: Top Two-Column Analyzer Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              <div className="lg:col-span-7">
                <ConversationInput
                  conversationText={conversationText}
                  setConversationText={setConversationText}
                  onAnalyze={() => handleAnalyze()}
                  onClear={handleClear}
                  isLoading={isLoading}
                  activeStage={activeStage}
                />
              </div>

              <div className="lg:col-span-5">
                <AnalysisSummary metrics={evalOutput?.metrics || null} isLoading={isLoading} />
              </div>
            </div>

            {/* Error State */}
            {error && (
              <ErrorState errorMessage={error} onRetry={() => handleAnalyze()} onClear={handleClear} />
            )}

            {/* Empty State */}
            {!evalOutput && !isLoading && !error && (
              <EmptyState onLoadExample={handleLoadExample} />
            )}

            {/* PAGE 2: Facet Results Table / Card List */}
            {evalOutput && !isLoading && (
              <section className="pt-2 space-y-6">
                {/* Retrieval Inspector Banner */}
                {evalOutput.retrieved_facets && evalOutput.retrieved_facets.length > 0 && (
                  <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-xs">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 uppercase tracking-wider">
                            Retrieval Subsetting
                          </span>
                          <span className="text-xs text-slate-500 font-medium">
                            {evalOutput.retrieved_facets.length} of 399 catalogue facets selected
                          </span>
                        </div>
                        <h3 className="text-sm font-bold text-slate-900 mt-1">
                          Retrieved Candidate Subset
                        </h3>
                      </div>
                      <p className="text-xs text-slate-500 max-w-sm sm:text-right">
                        Candidate facets pruned via FAISS dense embeddings & intent routing prior to scoring.
                      </p>
                    </div>

                    <div className="flex items-center flex-wrap gap-1.5 pt-1">
                      {evalOutput.retrieved_facets.map((rf, idx) => (
                        <button
                          key={idx}
                          type="button"
                          onClick={() => setSelectedFacetName(rf.facet)}
                          className="inline-flex items-center space-x-1 px-2.5 py-1 rounded-md bg-slate-50 hover:bg-slate-100 border border-slate-200 text-[11px] text-slate-800 transition-colors"
                        >
                          <span className="font-semibold">{rf.facet}</span>
                          <span className="text-[10px] font-mono text-slate-400">({rf.similarity})</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-extrabold text-slate-900">Facet Evaluation Results</h2>
                    <p className="text-xs text-slate-500">
                      Detailed status, score levels, evidence quotes, and abstention reasoning for candidate facets.
                    </p>
                  </div>
                </div>

                {/* Filter & Search Bar */}
                <FacetFilterBar
                  activeFilter={activeFilter}
                  setActiveFilter={setActiveFilter}
                  searchQuery={searchQuery}
                  setSearchQuery={setSearchQuery}
                  counts={counts}
                />

                {/* Results Card List */}
                <FacetResultsList
                  results={filteredResults}
                  onSelectFacet={(facetName) => setSelectedFacetName(facetName)}
                />
              </section>
            )}
          </div>
        )}

        {/* PAGE 3: System Overview */}
        {activeTab === 'overview' && <SystemOverview />}

        {/* PAGE 4: Benchmark Suite */}
        {activeTab === 'benchmark' && (
          <BenchmarkPage onSelectSampleText={handleSelectSampleText} />
        )}
      </main>

      {/* Facet Detail Side Panel Modal */}
      {selectedFacetName && (
        <FacetDetailModal
          facetName={selectedFacetName}
          currentResult={currentResultObject}
          onClose={() => setSelectedFacetName(null)}
        />
      )}

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 mt-12 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>FacetLens — Internal ML Evaluation Tool</span>
          <span className="text-slate-400">Zero-Hallucination Evidence-Grounded Scoring Pipeline</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
