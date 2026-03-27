import React, { useState, useEffect } from 'react';
import { RefreshCw, Filter, Sparkles, AlertCircle, Info, ArrowRight, Heart } from 'lucide-react';
import PaperCard from './components/PaperCard';

interface Paper {
  id: string;
  title: string;
  authors: string[];
  summary: string;
  published: string;
  url: string;
  tags: string[];
}

interface DigestResponse {
  date: string;
  papers: Paper[];
  error?: string;
}

const App: React.FC = () => {
  const [digest, setDigest] = useState<DigestResponse | null>(null);
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showOnlyFavorites, setShowOnlyFavorites] = useState(false);
  const [favoriteIds, setFavoriteIds] = useState<string[]>([]);

  useEffect(() => {
    fetchDigest();
    updateFavoriteIds();
  }, []);

  const updateFavoriteIds = () => {
    const favorites = JSON.parse(localStorage.getItem('dl_digest_favorites') || '[]');
    setFavoriteIds(favorites);
  };

  const fetchDigest = async () => {
    setIsLoading(true);
    try {
      // Fetch the static JSON file from public folder
      const res = await fetch('./digest.json');
      if (!res.ok) throw new Error('Digest not found');
      const data = await res.json();
      setDigest(data);
      
      // Calculate available tags from the data itself
      const tags = new Set<string>();
      data.papers.forEach((p: Paper) => p.tags.forEach(t => tags.add(t)));
      setAvailableTags(Array.from(tags).sort());

    } catch (err) {
      console.error('Error fetching digest:', err);
      setDigest({ date: '', papers: [], error: 'No digest available. Run update_digest.py' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleTagToggle = (tag: string) => {
    const nextTags = selectedTags.includes(tag)
      ? selectedTags.filter(t => t !== tag)
      : [...selectedTags, tag];
    setSelectedTags(nextTags);
  };

  // Logic moved from backend to frontend for static deployment
  const filteredPapers = digest?.papers?.filter(paper => {
    // 1. Filter by favorites if enabled
    if (showOnlyFavorites && !favoriteIds.includes(paper.id)) {
      return false;
    }
    
    // 2. Filter by tags (Intersection/ALL)
    if (selectedTags.length > 0) {
      return selectedTags.every(tag => paper.tags.includes(tag));
    }
    
    return true;
  });

  return (
    <div className="min-h-screen bg-[#F8FAFC]">
      {/* High-End Header */}
      <header className="bg-white border-b border-slate-200/60 sticky top-0 z-50 backdrop-blur-md bg-white/80">
        <div className="max-w-7xl mx-auto px-6 h-24 flex items-center justify-between">
          <div className="flex flex-col">
            <h1 className="text-3xl font-black tracking-tighter text-slate-900 flex items-center gap-3">
              <span className="bg-indigo-600 text-white px-3 py-1 rounded-xl shadow-lg shadow-indigo-200">DL</span> 
              <span className="tracking-tight">THEORY DIGEST</span>
            </h1>
          </div>
          
          <div className="hidden md:flex items-center gap-8">
            <div className="h-12 w-[1px] bg-slate-100" />
            <div className="text-right">
              <div className="text-[10px] font-black uppercase tracking-widest text-slate-400 mb-1">Issue Date</div>
              <div className="text-sm font-black text-slate-900 tabular-nums">{digest?.date || '--'}</div>
            </div>
            <button 
              onClick={() => fetchDigest()}
              className="p-3 bg-slate-50 text-slate-400 hover:text-indigo-600 hover:bg-white hover:shadow-xl transition-all rounded-2xl"
            >
              <RefreshCw size={20} className={isLoading ? 'animate-spin' : ''} />
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="bg-slate-900 text-white py-12 px-6 overflow-hidden relative">
        <div className="max-w-7xl mx-auto relative z-10">
          <div className="max-w-2xl">
            <div className="flex items-center gap-4 text-xs font-bold uppercase tracking-widest">
              <span className="flex items-center gap-2 text-emerald-400 bg-emerald-400/10 px-3 py-1.5 rounded-full border border-emerald-400/20">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Updated {digest?.date}
              </span>
              <span className="text-slate-500">•</span>
              <span className="text-slate-300">
                {digest?.papers?.length || 0} new papers since {
                  digest?.date 
                    ? new Date(new Date(digest.date).getTime() - 7 * 24 * 60 * 60 * 1000).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
                    : '--'
                }
              </span>
            </div>
          </div>
        </div>
        {/* Background Decorative Element */}
        <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-indigo-500/10 to-transparent pointer-events-none" />
        <div className="absolute -bottom-24 -right-24 w-96 h-96 bg-indigo-600/20 rounded-full blur-[120px] pointer-events-none" />
      </section>

      <main className="max-w-7xl mx-auto px-6 py-16">
        <div className="flex flex-col lg:flex-row gap-16">
          {/* Minimalist Navigation */}
          <aside className="w-full lg:w-72 space-y-12">
            <div>
              <h3 className="text-[11px] font-black uppercase tracking-[0.3em] text-slate-400 mb-8 flex items-center gap-3">
                <Filter size={14} className="text-indigo-500" /> Filter Research
              </h3>

              {/* Favorites Toggle */}
              <button
                onClick={() => {
                  updateFavoriteIds();
                  setShowOnlyFavorites(!showOnlyFavorites);
                }}
                className={`w-full group px-5 py-4 text-xs text-left transition-all font-black uppercase tracking-wider rounded-2xl border-2 flex items-center justify-between mb-8 ${
                  showOnlyFavorites
                    ? 'bg-rose-500 border-rose-500 text-white shadow-2xl shadow-rose-200'
                    : 'bg-white border-slate-100 text-slate-500 hover:border-rose-200 hover:text-rose-500'
                }`}
              >
                <span className="flex items-center gap-3">
                  <Heart size={14} fill={showOnlyFavorites ? "currentColor" : "none"} />
                  Saved Research
                </span>
                <div className={`w-2 h-2 rounded-full ${showOnlyFavorites ? 'bg-white animate-pulse' : 'bg-rose-200'}`} />
              </button>

              <div className="grid grid-cols-1 gap-3">
                {availableTags.map(tag => (
                  <button
                    key={tag}
                    onClick={() => handleTagToggle(tag)}
                    className={`group px-5 py-4 text-xs text-left transition-all font-black uppercase tracking-wider rounded-2xl border-2 flex items-center justify-between ${
                      selectedTags.includes(tag)
                        ? 'bg-slate-900 border-slate-900 text-white shadow-2xl shadow-slate-200'
                        : 'bg-white border-transparent text-slate-500 hover:border-slate-200 hover:text-slate-900'
                    }`}
                  >
                    <span className="flex items-center gap-3">
                      {tag === 'wildcard' && <Sparkles size={14} className="text-indigo-400" />}
                      {tag}
                    </span>
                    <ArrowRight size={14} className={`transition-transform duration-300 ${selectedTags.includes(tag) ? 'translate-x-0 opacity-100' : '-translate-x-2 opacity-0'}`} />
                  </button>
                ))}
              </div>
            </div>

            <div className="p-8 bg-indigo-600 rounded-[32px] text-white shadow-2xl shadow-indigo-200 relative overflow-hidden group">
              <div className="relative z-10">
                <Info size={24} className="mb-4 text-indigo-200" />
                <h4 className="text-lg font-black tracking-tight mb-2 uppercase">Manual Update</h4>
                <p className="text-xs text-indigo-100 leading-relaxed font-bold opacity-80 uppercase tracking-widest">
                  Run update_digest.py to pull the latest papers from arXiv.
                </p>
              </div>
              <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -translate-y-1/2 translate-x-1/2 group-hover:scale-125 transition-transform duration-500" />
            </div>
          </aside>

          {/* Paper Grid */}
          <div className="flex-1">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-48 text-slate-400">
                <div className="relative w-16 h-16 mb-8">
                  <div className="absolute inset-0 border-4 border-slate-100 rounded-full" />
                  <div className="absolute inset-0 border-4 border-t-indigo-500 rounded-full animate-spin" />
                </div>
                <p className="text-xs font-black uppercase tracking-[0.4em] text-slate-500 animate-pulse">Sourcing Fundamental Research</p>
              </div>
            ) : digest?.error ? (
              <div className="flex flex-col items-center justify-center py-32 bg-white border-2 border-dashed border-slate-200 rounded-[40px]">
                <AlertCircle size={48} className="mb-6 text-slate-200" />
                <p className="font-black uppercase tracking-widest text-slate-400">{digest.error}</p>
              </div>
            ) : (
              <div className="space-y-10">
                {filteredPapers?.map(paper => (
                  <PaperCard key={paper.id} paper={paper} />
                ))}
                
                {filteredPapers?.length === 0 && (
                  <div className="py-48 text-center bg-white border-2 border-dashed border-slate-200 rounded-[40px]">
                    <p className="text-xs font-black uppercase tracking-[0.4em] text-slate-300">
                      {showOnlyFavorites ? 'No Saved Papers Found' : 'Empty Set — Adjust Filters'}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="max-w-7xl mx-auto px-6 py-20 border-t border-slate-200/60 mt-20">
        <div className="flex flex-col md:flex-row justify-between items-center gap-8">
          <h2 className="text-xl font-black tracking-tighter text-slate-300">
            DL THEORY DIGEST
          </h2>
          <p className="text-[10px] font-black uppercase tracking-[0.4em] text-slate-300">
            &copy; 2026 Academic Research Compiled &bull; UIUC Statistics
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
