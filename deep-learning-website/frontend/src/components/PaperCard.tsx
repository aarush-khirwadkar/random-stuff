import React, { useState, useEffect } from 'react';
import { ExternalLink, ChevronDown, ChevronUp, Sparkles, Calendar, User, Heart } from 'lucide-react';
import { InlineMath } from 'react-katex';

interface PaperProps {
  paper: {
    id: string;
    title: string;
    authors: string[];
    summary: string;
    published: string;
    url: string;
    tags: string[];
  };
}

const TAG_STYLES: Record<string, string> = {
  'optimization': 'bg-blue-50 text-blue-700 border-blue-200',
  'generalization': 'bg-emerald-50 text-emerald-700 border-emerald-200',
  'approximation': 'bg-purple-50 text-purple-700 border-purple-200',
  'implicit bias': 'bg-indigo-50 text-indigo-700 border-indigo-200',
  'interpolation': 'bg-cyan-50 text-cyan-700 border-cyan-200',
  'high dimensional statistics': 'bg-rose-50 text-rose-700 border-rose-200',
  'generative models': 'bg-amber-50 text-amber-700 border-amber-200',
  'ai safety': 'bg-slate-50 text-slate-800 border-slate-300',
  'wildcard': 'bg-yellow-100 text-yellow-800 border-yellow-300',
};

const RenderWithMath: React.FC<{ text: string }> = ({ text }) => {
  // Simple regex to split by $...$ (inline math)
  const parts = text.split(/(\$[^\$]+\$)/g);
  
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('$') && part.endsWith('$')) {
          const math = part.slice(1, -1);
          return <InlineMath key={i} math={math} />;
        }
        return <span key={i}>{part}</span>;
      })}
    </>
  );
};

const PaperCard: React.FC<PaperProps> = ({ paper }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isFavorited, setIsFavorited] = useState(false);

  useEffect(() => {
    const favorites = JSON.parse(localStorage.getItem('dl_digest_favorites') || '[]');
    setIsFavorited(favorites.includes(paper.id));
  }, [paper.id]);

  const toggleFavorite = () => {
    const favorites = JSON.parse(localStorage.getItem('dl_digest_favorites') || '[]');
    let nextFavorites;
    if (favorites.includes(paper.id)) {
      nextFavorites = favorites.filter((id: string) => id !== paper.id);
    } else {
      nextFavorites = [...favorites, paper.id];
    }
    localStorage.setItem('dl_digest_favorites', JSON.stringify(nextFavorites));
    setIsFavorited(!isFavorited);
  };

  const summaryPreview = paper.summary.slice(0, 200).trim() + "...";

  return (
    <div className="group bg-white rounded-2xl border border-slate-200 p-8 shadow-sm hover:shadow-xl hover:border-indigo-100 transition-all duration-300 relative">
      <div className="flex justify-between items-start mb-6">
        <div className="flex flex-wrap gap-2">
          {paper.tags.map(tag => (
            <span 
              key={tag} 
              className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest border shadow-xs transition-transform hover:scale-105 ${
                  TAG_STYLES[tag] || 'bg-slate-50 text-slate-500 border-slate-200'
              }`}
            >
              {tag === 'wildcard' && <Sparkles size={10} className="inline mr-1" />}
              {tag}
            </span>
          ))}
        </div>

        <button 
          onClick={toggleFavorite}
          className={`p-3 rounded-full transition-all duration-300 ${
            isFavorited 
              ? 'bg-rose-50 text-rose-500 scale-110 shadow-lg shadow-rose-100' 
              : 'bg-slate-50 text-slate-300 hover:text-rose-400 hover:bg-rose-50'
          }`}
          title={isFavorited ? "Remove from Favorites" : "Add to Favorites"}
        >
          <Heart size={20} fill={isFavorited ? "currentColor" : "none"} strokeWidth={isFavorited ? 0 : 2} />
        </button>
      </div>

      <h3 className="text-2xl md:text-3xl font-black text-slate-900 mb-4 leading-[1.1] tracking-tight group-hover:text-indigo-600 transition-colors">
        {paper.title}
      </h3>
      
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-[11px] text-slate-400 mb-8 font-bold uppercase tracking-wider">
        <div className="flex items-center gap-2">
          <User size={14} className="text-indigo-400" />
          <span className="text-slate-600">{paper.authors.slice(0, 3).join(', ')}{paper.authors.length > 3 ? ' et al.' : ''}</span>
        </div>
        <div className="flex items-center gap-2">
          <Calendar size={14} className="text-indigo-400" />
          <span className="text-slate-600">{new Date(paper.published).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</span>
        </div>
      </div>

      <div className="text-slate-600 text-base leading-relaxed mb-8 font-medium">
        {isExpanded ? (
          <div className="space-y-4 animate-in fade-in slide-in-from-top-4 duration-500">
            {paper.summary.split('\n').map((para, i) => (
              <p key={i}><RenderWithMath text={para} /></p>
            ))}
          </div>
        ) : (
          <p className="opacity-80"><RenderWithMath text={summaryPreview} /></p>
        )}
      </div>

      <div className="flex items-center gap-6">
        <a 
          href={paper.url} 
          target="_blank" 
          rel="noopener noreferrer"
          className="inline-flex items-center gap-3 px-6 py-3 bg-slate-900 text-white text-xs font-black uppercase tracking-[0.2em] hover:bg-indigo-600 hover:-translate-y-1 transition-all rounded-full shadow-lg shadow-indigo-100"
        >
          Open on arXiv <ExternalLink size={14} />
        </a>
        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          className="inline-flex items-center gap-2 text-xs font-black uppercase tracking-[0.2em] text-slate-400 hover:text-indigo-600 transition-colors"
        >
          {isExpanded ? 'Minimize' : 'Read Abstract'}
          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>
    </div>
  );
};

export default PaperCard;
