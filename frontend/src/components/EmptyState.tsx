import React from 'react';
import { Layers, Sparkles } from 'lucide-react';

interface EmptyStateProps {
  onLoadExample: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ onLoadExample }) => {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-12 text-center shadow-xs max-w-2xl mx-auto my-6">
      <div className="w-14 h-14 rounded-2xl bg-slate-100 border border-slate-200 flex items-center justify-center mx-auto mb-4 text-slate-700 shadow-2xs">
        <Layers className="w-7 h-7" />
      </div>

      <h2 className="text-lg font-bold text-slate-900 tracking-tight">Analyze a Conversation</h2>
      <p className="text-xs text-slate-600 max-w-md mx-auto mt-1.5 leading-relaxed">
        Enter a conversation snippet above to retrieve relevant behavioral facets and evaluate only what the available conversational evidence strictly supports.
      </p>

      <div className="mt-6 flex justify-center">
        <button
          onClick={onLoadExample}
          className="inline-flex items-center space-x-2 px-4 py-2 bg-slate-900 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 transition-colors shadow-xs"
        >
          <Sparkles className="w-3.5 h-3.5 text-amber-300" />
          <span>Load Standard Sample Snippet</span>
        </button>
      </div>
    </div>
  );
};
