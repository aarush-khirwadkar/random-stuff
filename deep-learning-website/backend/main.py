from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="DL Theory Digest API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def get_latest_digest_file():
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("digest_") and f.endswith(".json")]
    if not files:
        return None
    files.sort(reverse=True)
    return os.path.join(DATA_DIR, files[0])

@app.get("/api/digest")
async def get_digest(tags: Optional[List[str]] = Query(None)):
    digest_file = get_latest_digest_file()
    if not digest_file:
        return {"error": "No digest available. Run the update script."}
        
    with open(digest_file, "r") as f:
        papers = json.load(f)
        
    if tags:
        papers = [p for p in papers if all(tag in p["tags"] for tag in tags)]
        
    return {
        "date": digest_file.split("_")[1].replace(".json", ""),
        "papers": papers
    }

@app.get("/api/available_tags")
async def get_tags():
    from fetcher import TAG_KEYWORDS
    tags = list(TAG_KEYWORDS.keys()) + ["wildcard"]
    return sorted(tags)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
