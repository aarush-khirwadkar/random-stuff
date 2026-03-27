import arxiv
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict

# Tags and their associated keywords for categorization
TAG_KEYWORDS = {
    "optimization": ["optimization", "gradient descent", "convergence", "adam", "sgd", "loss landscape", "pl-condition"],
    "generalization": ["generalization", "rademacher complexity", "vc dimension", "stability", "excess risk", "out-of-distribution"],
    "approximation": ["approximation", "universal approximation", "expressivity", "depth-width trade-off", "complexity theory"],
    "implicit bias": ["implicit bias", "gradient flow", "margin maximization", "over-parameterization", "inductive bias"],
    "interpolation": ["interpolation", "double descent", "benign overfitting", "kernel regime", "ntk", "neural tangent kernel"],
    "high dimensional statistics": ["high dimensional", "sparsity", "compressed sensing", "random matrix theory", "concentration of measure"],
    "generative models": ["generative models", "gan", "vae", "diffusion", "score-based", "flow-based"],
    "ai safety": ["alignment", "robustness", "adversarial", "interpretability", "fairness", "mechanistic interpretability"]
}

CATEGORIES = ["stat.ML", "cs.LG", "math.ST", "math.OC", "cs.IT"]

def categorize_paper(title: str, abstract: str) -> List[str]:
    text = (title + " " + abstract).lower()
    tags = []
    for tag, keywords in TAG_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return tags

def get_wildcard_paper(papers: List[Dict]) -> Dict:
    # Logic to find a "Wildcard" paper
    for paper in papers:
        if "fundamental" in paper["summary"].lower() or "novel" in paper["summary"].lower():
            if len(paper["tags"]) <= 1: 
                return paper
    return papers[0] if papers else None

def get_latest_digest():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        return None, []
    files = [f for f in os.listdir(data_dir) if f.startswith("digest_") and f.endswith(".json")]
    if not files:
        return None, []
    files.sort(reverse=True)
    latest_path = os.path.join(data_dir, files[0])
    with open(latest_path, "r") as f:
        return latest_path, json.load(f)

def fetch_weekly_papers(favorited_ids: List[str] = []):
    client = arxiv.Client()
    
    # Calculate date range for the last 7 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    query_string = " OR ".join([f"cat:{cat}" for cat in CATEGORIES])
    
    search = arxiv.Search(
        query=query_string,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results = []
    for result in client.results(search):
        # We only want papers from the last week
        if result.published.replace(tzinfo=None) < start_date:
            break
            
        tags = categorize_paper(result.title, result.summary)
        
        if tags:
            results.append({
                "id": result.entry_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat(),
                "url": result.pdf_url,
                "tags": tags
            })

    # If a favorited paper is not in results, keep it
    current_ids = {p["id"] for p in results}
    ids_to_fetch = [fid for fid in favorited_ids if fid not in current_ids]
    
    if ids_to_fetch:
        id_search = arxiv.Search(id_list=ids_to_fetch)
        for result in client.results(id_search):
            tags = categorize_paper(result.title, result.summary)
            results.append({
                "id": result.entry_id,
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "published": result.published.isoformat(),
                "url": result.pdf_url,
                "tags": tags
            })
            
    wildcard = get_wildcard_paper(results)
    if wildcard:
        wildcard["tags"].append("wildcard")
        
    return results

def save_digest(papers: List[Dict]):
    date_str = datetime.now().strftime("%Y-%m-%d")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Format for the frontend
    digest_data = {
        "date": date_str,
        "papers": papers
    }
    
    # 1. Save to archive
    file_path = os.path.join(data_dir, f"digest_{date_str}.json")
    with open(file_path, "w") as f:
        json.dump(papers, f, indent=4) # Archive keeps raw list for compatibility
    
    # 2. Save to frontend public folder for static deployment
    public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "public")
    os.makedirs(public_dir, exist_ok=True)
    public_path = os.path.join(public_dir, "digest.json")
    with open(public_path, "w") as f:
        json.dump(digest_data, f, indent=4)
        
    print(f"Saved {len(papers)} papers to {file_path} and {public_path}")

if __name__ == "__main__":
    # In a real scenario, you'd pass favorited IDs here.
    # For now, it defaults to empty to avoid breaking the manual run.
    papers = fetch_weekly_papers()
    save_digest(papers)
