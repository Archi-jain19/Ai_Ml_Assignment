import React from 'react';
import { Search } from 'lucide-react';

export type FilterStatus = 'all' | 'scored' | 'insufficient_evidence' | 'not_observable';

interface FacetFilterBarProps {
  activeFilter: FilterStatus;
  setActiveFilter: (filter: FilterStatus) => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  counts: {
    all: number;
    scored: number;
    insufficient_evidence: number;
    not_observable: number;
  };
}

export const FacetFilterBar: React.FC<FacetFilterBarProps> = ({
  activeFilter,
  setActiveFilter,
  searchQuery,
  setSearchQuery,
  counts,
}) => {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-xs mb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      {/* Filter Tabs */}
      <div className="flex items-center flex-wrap gap-1.5">
        <button
          onClick={() => setActiveFilter('all')}
          className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeFilter === 'all'
              ? 'bg-slate-900 text-white shadow-xs'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200/70'
          }`}
        >
          <span>All</span>
          <span className={`px-1.5 py-0.2 rounded-full text-[10px] ${activeFilter === 'all' ? 'bg-slate-700 text-white' : 'bg-slate-200 text-slate-700'}`}>
            {counts.all}
          </span>
        </button>

        <button
          onClick={() => setActiveFilter('scored')}
          className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeFilter === 'scored'
              ? 'bg-emerald-700 text-white shadow-xs'
              : 'bg-emerald-50 text-emerald-800 hover:bg-emerald-100 border border-emerald-200'
          }`}
        >
          <span>Scored</span>
          <span className={`px-1.5 py-0.2 rounded-full text-[10px] ${activeFilter === 'scored' ? 'bg-emerald-900 text-white' : 'bg-emerald-100 text-emerald-800'}`}>
            {counts.scored}
          </span>
        </button>

        <button
          onClick={() => setActiveFilter('insufficient_evidence')}
          className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeFilter === 'insufficient_evidence'
              ? 'bg-amber-700 text-white shadow-xs'
              : 'bg-amber-50 text-amber-800 hover:bg-amber-100 border border-amber-200'
          }`}
        >
          <span>Insufficient Evidence</span>
          <span className={`px-1.5 py-0.2 rounded-full text-[10px] ${activeFilter === 'insufficient_evidence' ? 'bg-amber-900 text-white' : 'bg-amber-100 text-amber-800'}`}>
            {counts.insufficient_evidence}
          </span>
        </button>

        <button
          onClick={() => setActiveFilter('not_observable')}
          className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            activeFilter === 'not_observable'
              ? 'bg-slate-700 text-white shadow-xs'
              : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
          }`}
        >
          <span>Not Observable</span>
          <span className={`px-1.5 py-0.2 rounded-full text-[10px] ${activeFilter === 'not_observable' ? 'bg-slate-800 text-white' : 'bg-slate-200 text-slate-700'}`}>
            {counts.not_observable}
          </span>
        </button>
      </div>

      {/* Search Input */}
      <div className="relative min-w-[240px]">
        <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5 pointer-events-none" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search facets..."
          className="w-full pl-9 pr-3 py-1.5 text-xs text-slate-900 bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-900 focus:bg-white transition-all font-medium"
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery('')}
            className="absolute right-2.5 top-2 text-xs text-slate-400 hover:text-slate-700 font-bold"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
};
