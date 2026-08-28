import React from 'react';
import { Play, RotateCcw, ArrowRight } from 'lucide-react';

interface ConversationInputProps {
  conversationText: string;
  setConversationText: (text: string) => void;
  onAnalyze: () => void;
  onClear: () => void;
  isLoading: boolean;
  activeStage?: 'retrieval' | 'scoring' | 'validation' | null;
}

export const ConversationInput: React.FC<ConversationInputProps> = ({
  conversationText,
  setConversationText,
  onAnalyze,
  onClear,
  isLoading,
  activeStage,
}) => {
  const charCount = conversationText.length;
  const wordCount = conversationText.trim() ? conversationText.trim().split(/\s+/).length : 0;

  const presets = [
    {
      label: 'Temporal Workflow',
      text: 'I used to miss deadlines. I changed my workflow, created a weekly plan, followed it for six months, and now finish projects three days early.',
    },
    {
      label: 'Medical Trap (Glucose)',
      text: 'I had a blood test last week and my glucose was normal.',
    },
    {
      label: 'Quoted Hostility',
      text: 'My manager walked in and screamed, "You are all completely incompetent!" I just kept my voice calm, took notes, and asked what priorities we should adjust.',
    },
    {
      label: 'Third-Party Trait',
      text: 'My friend is extremely patient and my manager is very organized.',
    },
    {
      label: 'Sarcasm',
      text: 'Oh, wonderful! Another unexpected 7 AM production outage. Truly the highlight of my week.',
    },
  ];

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-xs flex flex-col h-full">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Conversation Input</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Enter a conversation snippet to evaluate against candidate facets.
          </p>
        </div>
      </div>

      {/* Quick Test Presets */}
      <div className="mb-3">
        <div className="flex items-center space-x-1.5 overflow-x-auto pb-1 text-xs">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider shrink-0 mr-1">Presets:</span>
          {presets.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => setConversationText(p.text)}
              className="px-2.5 py-1 rounded-md bg-slate-100 hover:bg-slate-200/80 text-slate-700 font-medium text-[11px] shrink-0 transition-colors border border-slate-200"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Textarea */}
      <div className="flex-1 mb-3">
        <textarea
          value={conversationText}
          onChange={(e) => setConversationText(e.target.value)}
          placeholder={`Example: ${presets[0].text}`}
          rows={7}
          disabled={isLoading}
          className="w-full h-full min-h-[160px] p-3 text-sm text-slate-800 placeholder-slate-400 bg-slate-50/50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-900 focus:border-transparent transition-all resize-y font-normal"
        />
      </div>

      {/* Character Count & Info */}
      <div className="flex items-center justify-between text-xs text-slate-500 mb-4 px-1">
        <span>{wordCount} words</span>
        <span>{charCount} characters</span>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center space-x-3 mb-4">
        <button
          type="button"
          onClick={onAnalyze}
          disabled={isLoading || !conversationText.trim()}
          className="flex-1 inline-flex items-center justify-center space-x-2 px-4 py-2.5 bg-slate-900 text-white rounded-lg text-sm font-medium hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-900 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-xs"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              <span>Evaluating Evidence...</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 text-white fill-white" />
              <span>Analyze Conversation</span>
            </>
          )}
        </button>

        <button
          type="button"
          onClick={onClear}
          disabled={isLoading || (!conversationText && !activeStage)}
          className="inline-flex items-center space-x-1.5 px-4 py-2.5 bg-white text-slate-700 border border-slate-200 rounded-lg text-sm font-medium hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-slate-400 transition-all"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
          <span>Clear</span>
        </button>
      </div>

      {/* Pipeline Indicator */}
      <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
        <span className="font-semibold text-slate-400 uppercase tracking-wider text-[10px]">Pipeline</span>
        <div className="flex items-center space-x-1.5 font-medium">
          <span className={`px-1.5 py-0.5 rounded ${activeStage === 'retrieval' ? 'bg-slate-900 text-white font-semibold' : 'bg-slate-100 text-slate-600'}`}>
            Retrieval
          </span>
          <ArrowRight className="w-3 h-3 text-slate-300" />
          <span className={`px-1.5 py-0.5 rounded ${activeStage === 'scoring' ? 'bg-slate-900 text-white font-semibold' : 'bg-slate-100 text-slate-600'}`}>
            Routing
          </span>
          <ArrowRight className="w-3 h-3 text-slate-300" />
          <span className={`px-1.5 py-0.5 rounded ${activeStage === 'scoring' ? 'bg-slate-900 text-white font-semibold' : 'bg-slate-100 text-slate-600'}`}>
            Scoring
          </span>
          <ArrowRight className="w-3 h-3 text-slate-300" />
          <span className={`px-1.5 py-0.5 rounded ${activeStage === 'validation' ? 'bg-slate-900 text-white font-semibold' : 'bg-slate-100 text-slate-600'}`}>
            Abstention
          </span>
        </div>
      </div>
    </div>
  );
};
