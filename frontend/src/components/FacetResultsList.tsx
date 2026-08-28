import React, { useState } from 'react';
import type { FacetResult } from '../types';
import { ChevronDown, ChevronUp, Info, CheckCircle2, AlertCircle, EyeOff } from 'lucide-react';

interface FacetResultsListProps {
  results: FacetResult[];
  onSelectFacet: (facetName: string) => void;
}

export const FacetResultsList: React.FC<FacetResultsListProps> = ({ results, onSelectFacet }) => {
  const [expandedFacet, setExpandedFacet] = useState<string | null>(null);

  const toggleExpand = (facet: string) => {
    setExpandedFacet((prev) => (prev === facet ? null : facet));
  };

  if (results.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-8 text-center shadow-xs">
        <AlertCircle className="w-8 h-8 text-slate-400 mx-auto mb-2 stroke-1" />
        <h3 className="text-sm font-semibold text-slate-800">No matching facets found</h3>
        <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
          Try adjusting your search query or status filters above.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {results.map((r) => {
        const isExpanded = expandedFacet === r.facet;
        const isScored = r.status === 'scored';
        const isInsufficient = r.status === 'insufficient_evidence';
        const isNotObservable = r.status === 'not_observable' || r.status === 'unsuitable';

        return (
          <div
            key={r.facet}
            className="bg-white rounded-xl border border-slate-200 shadow-xs hover:border-slate-300 transition-all overflow-hidden"
          >
            {/* Main Facet Header Row */}
            <div className="p-4 sm:p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
              {/* Left Column: Facet Name & Type Badge */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2.5 mb-1">
                  <h3
                    onClick={() => onSelectFacet(r.facet)}
                    className="text-base font-bold text-slate-900 hover:text-blue-600 cursor-pointer transition-colors truncate"
                  >
                    {r.facet}
                  </h3>
                  <button
                    onClick={() => onSelectFacet(r.facet)}
                    className="text-slate-400 hover:text-slate-600 p-0.5 rounded"
                    title="View Taxonomy Details"
                  >
                    <Info className="w-4 h-4" />
                  </button>
                </div>
                <p className="text-xs text-slate-500 line-clamp-1">{r.scoring_definition}</p>
              </div>

              {/* Status Badge */}
              <div className="shrink-0 flex items-center space-x-4">
                <div>
                  {isScored && (
                    <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                      <span>SCORED</span>
                    </span>
                  )}
                  {isInsufficient && (
                    <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-800 border border-amber-200">
                      <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
                      <span>INSUFFICIENT EVIDENCE</span>
                    </span>
                  )}
                  {isNotObservable && (
                    <span className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 border border-slate-200">
                      <EyeOff className="w-3.5 h-3.5 text-slate-500" />
                      <span>NOT OBSERVABLE</span>
                    </span>
                  )}
                </div>

                {/* Visual 5-Dot Score & Text Label */}
                <div className="w-36 text-center">
                  {isScored && r.score ? (
                    <div>
                      <div className="flex items-center justify-center space-x-1 text-emerald-600 mb-0.5">
                        {[1, 2, 3, 4, 5].map((dot) => (
                          <span
                            key={dot}
                            className={`w-2.5 h-2.5 rounded-full ${
                              dot <= (r.score || 0) ? 'bg-emerald-600' : 'bg-slate-200'
                            }`}
                          />
                        ))}
                        <span className="text-xs font-bold text-slate-900 ml-1.5">{r.score} / 5</span>
                      </div>
                      <span className="text-[11px] font-semibold text-slate-700 uppercase tracking-wide">
                        {r.score_label || 'High'}
                      </span>
                    </div>
                  ) : (
                    <div className="text-slate-400 font-mono text-sm font-semibold">
                      — <span className="text-[11px] font-sans font-normal text-slate-400 block">Abstained</span>
                    </div>
                  )}
                </div>

                {/* Confidence */}
                <div className="w-20 text-right">
                  <span className="text-xs font-semibold text-slate-900 block">
                    {Math.round(r.confidence * 100)}%
                  </span>
                  <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Conf</span>
                </div>

                {/* Expand Toggle Button */}
                <button
                  onClick={() => toggleExpand(r.facet)}
                  className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-lg transition-colors"
                >
                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Expandable Evidence & Reasoning Section */}
            {isExpanded && (
              <div className="bg-slate-50/70 border-t border-slate-100 p-4 sm:p-5 space-y-3 text-xs leading-relaxed">
                {isScored ? (
                  <>
                    {/* Evidence */}
                    <div>
                      <span className="font-bold text-slate-900 uppercase tracking-wider text-[10px] block mb-1">
                        Evidence
                      </span>
                      <blockquote className="p-2.5 bg-white border-l-3 border-emerald-500 rounded-r text-slate-800 italic shadow-2xs font-normal">
                        "{r.evidence || r.reason}"
                      </blockquote>
                    </div>

                    {/* Reasoning */}
                    <div>
                      <span className="font-bold text-slate-900 uppercase tracking-wider text-[10px] block mb-1">
                        Reasoning
                      </span>
                      <p className="text-slate-700 bg-white p-2.5 rounded border border-slate-200/70">
                        {r.reason}
                      </p>
                    </div>

                    {/* Metadata Footer */}
                    <div className="flex items-center justify-between pt-2 border-t border-slate-200/50 text-[11px] text-slate-500">
                      <span>Confidence Score: <strong className="text-slate-800">{r.confidence}</strong></span>
                      <button
                        onClick={() => onSelectFacet(r.facet)}
                        className="text-blue-600 hover:text-blue-800 font-semibold hover:underline"
                      >
                        Inspect Taxonomy Definition & Anchors →
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    {/* Abstention Explanation */}
                    <div>
                      <span className="font-bold text-amber-900 uppercase tracking-wider text-[10px] block mb-1">
                        Why We Abstained
                      </span>
                      <div className="p-3 bg-amber-50/60 border border-amber-200/70 rounded text-amber-900 leading-relaxed font-medium">
                        {r.reason}
                      </div>
                    </div>

                    <div>
                      <span className="font-bold text-slate-900 uppercase tracking-wider text-[10px] block mb-1">
                        Abstention Policy Rule
                      </span>
                      <p className="text-slate-600 bg-white p-2.5 rounded border border-slate-200">
                        {r.abstention_policy}
                      </p>
                    </div>

                    <div className="pt-1 flex items-center justify-end">
                      <button
                        onClick={() => onSelectFacet(r.facet)}
                        className="text-blue-600 hover:text-blue-800 text-[11px] font-semibold hover:underline"
                      >
                        Inspect Taxonomy Details →
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
