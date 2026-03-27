import sys
import os

# Add the backend directory to the path so we can import fetcher
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from fetcher import fetch_weekly_papers, save_digest, get_latest_digest

def update():
    print("Fetching weekly papers from arXiv...")
    
    # We can't access browser localStorage from here, 
    # but we can look at the latest digest file we generated
    # and see if there are any favorites there.
    # HOWEVER, the JSON doesn't store if it was favorited (that's local to the user).
    # To truly support "don't want removed", we'd need a way to pass these IDs.
    
    # For now, let's just make the script capable of accepting IDs if we ever want to automate it.
    papers = fetch_weekly_papers()
    
    if papers:
        save_digest(papers)
        print(f"Update complete! {len(papers)} papers found.")
    else:
        print("No papers matching the theory tags found this week.")

if __name__ == "__main__":
    update()
