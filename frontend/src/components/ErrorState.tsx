import React from 'react';
import { AlertOctagon, RefreshCw, RotateCcw } from 'lucide-react';

interface ErrorStateProps {
  errorMessage?: string | null;
  onRetry: () => void;
  onClear: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({ errorMessage, onRetry, onClear }) => {
  return (
    <div className="bg-red-50/70 border border-red-200 rounded-xl p-6 my-6 shadow-xs max-w-xl mx-auto text-center">
      <div className="w-12 h-12 rounded-full bg-red-100 border border-red-200 flex items-center justify-center mx-auto mb-3 text-red-600">
        <AlertOctagon className="w-6 h-6" />
      </div>

      <h3 className="text-base font-bold text-red-900">Analysis Failed</h3>
      <p className="text-xs text-red-700 mt-1 max-w-md mx-auto leading-relaxed">
        {errorMessage || 'The scoring service returned an invalid or incomplete response. Please try again.'}
      </p>

      <div className="flex items-center justify-center space-x-3 mt-5">
        <button
          onClick={onRetry}
          className="inline-flex items-center space-x-1.5 px-4 py-2 bg-red-600 text-white rounded-lg text-xs font-semibold hover:bg-red-700 transition-colors shadow-2xs"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Analysis</span>
        </button>

        <button
          onClick={onClear}
          className="inline-flex items-center space-x-1.5 px-4 py-2 bg-white text-slate-700 border border-slate-200 rounded-lg text-xs font-semibold hover:bg-slate-50 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
          <span>Clear Input</span>
        </button>
      </div>
    </div>
  );
};
