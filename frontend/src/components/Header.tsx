import React from 'react';
import { Layers, Activity, BarChart2, Cpu } from 'lucide-react';

export type PageTab = 'analyzer' | 'overview' | 'benchmark';

interface HeaderProps {
  activeTab: PageTab;
  setActiveTab: (tab: PageTab) => void;
  modelReady: boolean;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, modelReady }) => {
  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left Logo & Brand */}
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('analyzer')}>
              <div className="w-9 h-9 rounded-lg bg-slate-900 flex items-center justify-center text-white shadow-xs">
                <Layers className="w-5 h-5 text-slate-100" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-bold text-slate-900 text-lg tracking-tight">FacetLens</span>
                  <span className="text-xs font-semibold px-2 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                    ML Eval v1.0
                  </span>
                </div>
                <p className="text-xs text-slate-500 font-medium">Conversation Facet Evaluation</p>
              </div>
            </div>

            {/* Navigation Tabs */}
            <nav className="hidden md:flex items-center space-x-1 pl-4 border-l border-slate-200">
              <button
                onClick={() => setActiveTab('analyzer')}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'analyzer'
                    ? 'bg-slate-100 text-slate-900 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <Activity className="w-4 h-4 text-slate-500" />
                <span>Analyzer</span>
              </button>

              <button
                onClick={() => setActiveTab('overview')}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'overview'
                    ? 'bg-slate-100 text-slate-900 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <Cpu className="w-4 h-4 text-slate-500" />
                <span>System Overview</span>
              </button>

              <button
                onClick={() => setActiveTab('benchmark')}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'benchmark'
                    ? 'bg-slate-100 text-slate-900 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                }`}
              >
                <BarChart2 className="w-4 h-4 text-slate-500" />
                <span>Benchmark</span>
              </button>
            </nav>
          </div>

          {/* Right Status */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-50 border border-slate-200 text-xs font-medium text-slate-700">
              <span className={`relative flex h-2 w-2`}>
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${modelReady ? 'bg-emerald-400' : 'bg-amber-400'} opacity-75`}></span>
                <span className={`relative inline-flex rounded-full h-2 w-2 ${modelReady ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
              </span>
              <span>{modelReady ? 'Model Ready' : 'Connecting to Server...'}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
