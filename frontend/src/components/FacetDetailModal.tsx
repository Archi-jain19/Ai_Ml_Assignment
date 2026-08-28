import React, { useEffect, useState } from 'react';
import type { TaxonomyItem, FacetResult } from '../types';
import { X, ShieldAlert, CheckCircle2, AlertCircle, EyeOff, BookOpen } from 'lucide-react';

interface FacetDetailModalProps {
  facetName: string | null;
  currentResult?: FacetResult | null;
  onClose: () => void;
}

export const FacetDetailModal: React.FC<FacetDetailModalProps> = ({
  facetName,
  currentResult,
  onClose,
}) => {
  const [taxonomyData, setTaxonomyData] = useState<TaxonomyItem | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!facetName) return;

    setLoading(true);
    setError(null);

    fetch(`/api/facet/${encodeURIComponent(facetName)}`)
      .then((res) => {
        if (!res.ok) throw new Error('Facet taxonomy details not found.');
        return res.json();
      })
      .then((data) => setTaxonomyData(data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [facetName]);

  if (!facetName) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/40 backdrop-blur-xs flex justify-end transition-opacity">
      <div className="bg-white w-full max-w-xl min-h-screen shadow-2xl border-l border-slate-200 flex flex-col justify-between p-6 overflow-y-auto">
        <div>
          {/* Header */}
          <div className="flex items-start justify-between border-b border-slate-100 pb-4 mb-5">
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200 uppercase tracking-wider">
                  Taxonomy Entry
                </span>
                {taxonomyData?.sensitivity && (
                  <span className={`text-xs font-medium px-2 py-0.5 rounded border capitalize ${
                    taxonomyData.sensitivity === 'high' || taxonomyData.sensitivity === 'critical'
                      ? 'bg-red-50 text-red-700 border-red-200'
                      : 'bg-slate-50 text-slate-600 border-slate-200'
                  }`}>
                    {taxonomyData.sensitivity} Sensitivity
                  </span>
                )}
              </div>
              <h2 className="text-xl font-bold text-slate-900 mt-1">{facetName}</h2>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {loading ? (
            <div className="py-12 text-center text-slate-500">
              <div className="w-8 h-8 rounded-full border-2 border-slate-300 border-t-slate-900 animate-spin mx-auto mb-2"></div>
              <p className="text-xs font-medium">Loading taxonomy definition...</p>
            </div>
          ) : error ? (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-xs font-medium">
              {error}
            </div>
          ) : (
            <div className="space-y-6 text-xs text-slate-700">
              {/* Facet Metadata */}
              <div className="grid grid-cols-2 gap-3 bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                <div>
                  <span className="font-semibold text-slate-500 block uppercase tracking-wider text-[10px] mb-0.5">
                    Facet Type
                  </span>
                  <span className="font-bold text-slate-900 capitalize">
                    {taxonomyData?.facet_type.replace('_', ' ')}
                  </span>
                </div>
                <div>
                  <span className="font-semibold text-slate-500 block uppercase tracking-wider text-[10px] mb-0.5">
                    Conversational Observability
                  </span>
                  <span className={`font-bold ${taxonomyData?.conversation_observable ? 'text-emerald-700' : 'text-slate-600'}`}>
                    {taxonomyData?.conversation_observable ? 'Directly Observable' : 'Requires External Log/Instrument'}
                  </span>
                </div>
              </div>

              {/* Current Result in Pipeline */}
              {currentResult && (
                <div className="p-4 rounded-lg bg-slate-900 text-white space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Current Pipeline Evaluation</span>
                    <span className="text-xs font-semibold text-slate-300">Conf: {Math.round(currentResult.confidence * 100)}%</span>
                  </div>
                  <div className="flex items-center justify-between pt-1">
                    <div className="flex items-center space-x-2">
                      {currentResult.status === 'scored' && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
                      {currentResult.status === 'insufficient_evidence' && <AlertCircle className="w-4 h-4 text-amber-400" />}
                      {currentResult.status === 'not_observable' && <EyeOff className="w-4 h-4 text-slate-400" />}
                      <span className="font-bold text-sm uppercase tracking-wide">{currentResult.status.replace('_', ' ')}</span>
                    </div>
                    {currentResult.score && (
                      <span className="text-base font-extrabold text-emerald-400">{currentResult.score} / 5 ({currentResult.score_label})</span>
                    )}
                  </div>
                  <p className="text-xs text-slate-300 pt-2 border-t border-slate-800 leading-relaxed font-normal">
                    {currentResult.reason}
                  </p>
                </div>
              )}

              {/* Score Definition */}
              <div>
                <span className="font-bold text-slate-900 uppercase tracking-wider text-[10px] block mb-1.5 flex items-center space-x-1">
                  <BookOpen className="w-3.5 h-3.5 text-slate-500" />
                  <span>Score Definition</span>
                </span>
                <p className="bg-slate-50 p-3 rounded-lg border border-slate-200 text-slate-800 leading-relaxed font-normal">
                  {taxonomyData?.scoring_definition}
                </p>
              </div>

              {/* Score Anchors */}
              <div>
                <span className="font-bold text-slate-900 uppercase tracking-wider text-[10px] block mb-2">
                  Level Anchors (1 to 5)
                </span>
                <div className="space-y-1.5">
                  {[1, 2, 3, 4, 5].map((lvl) => {
                    const key = `score_${lvl}_anchor` as keyof TaxonomyItem;
                    const val = taxonomyData?.[key];
                    if (!val) return null;
                    return (
                      <div key={lvl} className="flex items-start space-x-2 p-2 rounded bg-slate-50/70 border border-slate-100 text-[11px]">
                        <span className="font-bold px-1.5 py-0.5 rounded bg-slate-200 text-slate-800 text-[10px] shrink-0">
                          L{lvl}
                        </span>
                        <span className="text-slate-700 leading-snug">{val}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Abstention Policy */}
              <div>
                <span className="font-bold text-amber-900 uppercase tracking-wider text-[10px] block mb-1.5 flex items-center space-x-1">
                  <ShieldAlert className="w-3.5 h-3.5 text-amber-600" />
                  <span>Abstention Policy & Guardrails</span>
                </span>
                <div className="p-3 bg-amber-50/70 border border-amber-200 rounded-lg text-amber-900 leading-relaxed font-medium">
                  {taxonomyData?.abstention_policy}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="pt-4 border-t border-slate-100 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-900 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 transition-colors"
          >
            Close Panel
          </button>
        </div>
      </div>
    </div>
  );
};
