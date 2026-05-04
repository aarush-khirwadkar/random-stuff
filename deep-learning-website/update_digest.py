import sys
import os

# Add the backend directory to the path so we can import fetcher
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from fetcher import fetch_weekly_papers, save_digest, get_latest_digest

def update():
    print("Fetching weekly papers from arXiv...")
    
    try:
        papers = fetch_weekly_papers()
        
        if papers:
            save_digest(papers)
            print(f"Update complete! {len(papers)} new papers saved to digest.")
        else:
            print("No new papers matching the theory tags found this week.")
    except Exception as e:
        print(f"ERROR during update: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    update()
