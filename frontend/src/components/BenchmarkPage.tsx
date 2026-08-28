import React, { useEffect, useState } from 'react';
import type { BenchmarkResponse } from '../types';
import { CheckCircle2, AlertTriangle, XCircle, RefreshCw, ArrowUpRight, Play, Database } from 'lucide-react';

interface BenchmarkPageProps {
  onSelectSampleText: (text: string) => void;
}

export const BenchmarkPage: React.FC<BenchmarkPageProps> = ({ onSelectSampleText }) => {
  const [data, setData] = useState<BenchmarkResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeCategoryFilter, setActiveCategoryFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const fetchBenchmark = () => {
    setLoading(true);
    setError(null);
    fetch('/api/benchmark')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to load benchmark evaluation dataset.');
        return res.json();
      })
      .then((json: BenchmarkResponse) => setData(json))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchBenchmark();
  }, []);

  const filteredRows = (data?.comparison_rows || []).filter((row) => {
    const matchesCat = activeCategoryFilter === 'all' || row.conversation_type === activeCategoryFilter;
    const matchesQuery =
      !searchQuery ||
      row.facet.toLowerCase().includes(searchQuery.toLowerCase()) ||
      row.conversation_text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      row.human_reasoning.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCat && matchesQuery;
  });

  return (
    <div className="max-w-7xl mx-auto space-y-6 py-6">
      {/* Top Banner & Run Action */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200 uppercase tracking-wider">
              Human Evaluation Benchmark
            </span>
            <span className="text-xs text-slate-500 font-medium">15 Standard Test Cases</span>
          </div>
          <h1 className="text-xl font-bold text-slate-900 mt-1">Benchmark Suite & Metric Evaluation</h1>
          <p className="text-xs text-slate-600 mt-0.5">
            Evaluates model predictions against human reference labels to measure agreement, precision, and abstention behavior.
          </p>
        </div>

        <button
          onClick={fetchBenchmark}
          disabled={loading}
          className="inline-flex items-center space-x-2 px-4 py-2.5 bg-slate-900 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-900 disabled:opacity-50 transition-all shadow-xs shrink-0"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>{loading ? 'Running Benchmark Suite...' : 'Re-Run Benchmark Suite'}</span>
        </button>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-xs font-medium">
          {error}
        </div>
      )}

      {/* Summary Metrics Cards */}
      {data?.summary && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
            <div className="flex items-center justify-between text-xs font-medium text-slate-500 mb-1">
              <span>Overall Agreement</span>
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            </div>
            <p className="text-3xl font-extrabold text-slate-900 tracking-tight">
              {data.summary.agreement_percentage}%
            </p>
            <span className="text-[11px] text-slate-500 mt-1 block">Exact status & score agreement</span>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
            <div className="flex items-center justify-between text-xs font-medium text-slate-500 mb-1">
              <span>Correct Abstentions</span>
              <Database className="w-4 h-4 text-blue-600" />
            </div>
            <p className="text-3xl font-extrabold text-blue-700 tracking-tight">
              {data.summary.correct_abstentions_percentage}%
            </p>
            <span className="text-[11px] text-slate-500 mt-1 block">Refused unsupported facets</span>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
            <div className="flex items-center justify-between text-xs font-medium text-slate-500 mb-1">
              <span>Incorrect Scores</span>
              <AlertTriangle className="w-4 h-4 text-amber-600" />
            </div>
            <p className="text-3xl font-extrabold text-amber-700 tracking-tight">
              {data.summary.incorrect_scores_percentage}%
            </p>
            <span className="text-[11px] text-slate-500 mt-1 block">False positive hallucinated scores</span>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
            <div className="flex items-center justify-between text-xs font-medium text-slate-500 mb-1">
              <span>Incorrect Abstentions</span>
              <XCircle className="w-4 h-4 text-red-500" />
            </div>
            <p className="text-3xl font-extrabold text-slate-800 tracking-tight">
              {data.summary.incorrect_abstentions_percentage}%
            </p>
            <span className="text-[11px] text-slate-500 mt-1 block">False negative missed evidence</span>
          </div>
        </div>
      )}

      {/* Benchmark Sample Cases Carousel/Grid */}
      {data?.conversations && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs">
          <h3 className="text-sm font-bold text-slate-900 mb-3">Benchmark Conversation Test Samples</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {data.conversations.slice(0, 6).map((c) => (
              <div
                key={c.conversation_id}
                onClick={() => onSelectSampleText(c.text)}
                className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg hover:border-slate-400 hover:bg-slate-100/70 transition-all cursor-pointer group flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-200 text-slate-700 uppercase tracking-wider">
                      {c.type}
                    </span>
                    <ArrowUpRight className="w-3.5 h-3.5 text-slate-400 group-hover:text-slate-900 transition-colors" />
                  </div>
                  <p className="text-xs text-slate-800 font-normal line-clamp-3 italic">"{c.text}"</p>
                </div>
                <span className="text-[11px] font-semibold text-blue-600 mt-2 block group-hover:underline">
                  Analyze this snippet →
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Comparison Table Section */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
        {/* Table Filter Controls */}
        <div className="p-4 border-b border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-3 bg-slate-50/50">
          <div className="flex items-center space-x-2 overflow-x-auto py-1">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider shrink-0">Category:</span>
            {['all', 'clear', 'sarcastic', 'contradictory', 'quoted', 'code-switched', 'adversarial_medical'].map((cat) => (
              <button
                key={cat}
                onClick={() => setActiveCategoryFilter(cat)}
                className={`px-2.5 py-1 rounded text-xs font-semibold shrink-0 capitalize transition-all ${
                  activeCategoryFilter === cat
                    ? 'bg-slate-900 text-white'
                    : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-100'
                }`}
              >
                {cat.replace('_', ' ')}
              </button>
            ))}
          </div>

          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search benchmark table..."
            className="px-3 py-1.5 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-900 w-full md:w-64"
          />
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-100/70 border-b border-slate-200 text-slate-700 font-bold uppercase tracking-wider text-[10px]">
              <tr>
                <th className="p-3.5 pl-5">Conversation & Category</th>
                <th className="p-3.5">Target Facet</th>
                <th className="p-3.5">Expected Human Label</th>
                <th className="p-3.5">Model Prediction</th>
                <th className="p-3.5">Reasoning / Attribution</th>
                <th className="p-3.5 pr-5 text-right">Result</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50/70 transition-colors">
                  {/* Conversation & Category */}
                  <td className="p-3.5 pl-5 max-w-xs">
                    <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200 uppercase tracking-wider block w-max mb-1">
                      {row.conversation_type}
                    </span>
                    <p className="text-slate-800 line-clamp-2 italic text-[11px]">"{row.conversation_text}"</p>
                    <button
                      onClick={() => onSelectSampleText(row.conversation_text)}
                      className="text-[10px] font-semibold text-blue-600 hover:underline mt-1 inline-flex items-center space-x-1"
                    >
                      <Play className="w-2.5 h-2.5" />
                      <span>Test in Analyzer</span>
                    </button>
                  </td>

                  {/* Target Facet */}
                  <td className="p-3.5 font-bold text-slate-900">{row.facet}</td>

                  {/* Expected Human Label */}
                  <td className="p-3.5">
                    <span className="font-semibold text-slate-800 uppercase text-[10px]">
                      {row.expected_status.replace('_', ' ')}
                    </span>
                    {row.expected_score && (
                      <span className="block font-bold text-slate-900 text-xs mt-0.5">
                        {row.expected_score} / 5
                      </span>
                    )}
                  </td>

                  {/* Model Prediction */}
                  <td className="p-3.5">
                    <span className={`font-semibold uppercase text-[10px] ${
                      row.predicted_status === 'scored' ? 'text-emerald-700' : 'text-amber-800'
                    }`}>
                      {row.predicted_status.replace('_', ' ')}
                    </span>
                    {row.predicted_score && (
                      <span className="block font-bold text-slate-900 text-xs mt-0.5">
                        {row.predicted_score} / 5
                      </span>
                    )}
                  </td>

                  {/* Reasoning */}
                  <td className="p-3.5 max-w-sm">
                    <p className="text-slate-700 text-[11px] leading-snug line-clamp-2">
                      {row.predicted_reason}
                    </p>
                  </td>

                  {/* Result Badge */}
                  <td className="p-3.5 pr-5 text-right font-bold shrink-0">
                    {row.result_badge === 'agreement' && (
                      <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-[11px]">
                        <span>✓ Agreement</span>
                      </span>
                    )}
                    {row.result_badge === 'partial' && (
                      <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 text-[11px]">
                        <span>△ Partial score</span>
                      </span>
                    )}
                    {row.result_badge === 'error' && (
                      <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-200 text-[11px]">
                        <span>✕ Discrepancy</span>
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
