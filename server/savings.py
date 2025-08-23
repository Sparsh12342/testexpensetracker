# server/savings.py
import time
from typing import List, Dict
import feedparser

# Category -> RSS feeds (reliable, no brittle HTML scraping)
CATEGORY_SOURCES: Dict[str, List[str]] = {
    "Dining": [
        "https://www.reddit.com/r/frugal/.rss",
        "https://www.reddit.com/r/deals/.rss",
        "https://www.reddit.com/r/EatCheapAndHealthy/.rss",
    ],
    "Groceries": [
        "https://www.reddit.com/r/frugal/.rss",
        "https://www.reddit.com/r/deals/.rss",
    ],
    "Shopping": [
        "https://www.reddit.com/r/deals/.rss",
        "https://www.reddit.com/r/BuyItForLife/.rss",
        "https://www.reddit.com/r/frugalmalefashion/.rss",
    ],
    "Transportation": [
        "https://www.reddit.com/r/frugal/.rss",
    ],
    "Utilities": [
        "https://www.reddit.com/r/frugal/.rss",
    ],
    "Subscriptions": [
        "https://www.reddit.com/r/frugal/.rss",
        "https://www.reddit.com/r/deals/.rss",
    ],
    "Entertainment": [
        "https://www.reddit.com/r/deals/.rss",
    ],
    "*": [
        "https://www.reddit.com/r/frugal/.rss",
        "https://www.reddit.com/r/deals/.rss",
    ],
}

def _fetch_rss(url: str, limit: int = 25):
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries[:limit]:
            title = (getattr(e, "title", "") or "").strip()
            link = getattr(e, "link", "") or ""
            if title and link:
                items.append({"title": title, "link": link, "source": url})
        return items
    except Exception:
        return []

def get_savings_suggestions(categories: List[str], merchants: List[str], max_items: int = 15):
    # choose sources for top categories
    srcs: List[str] = []
    for c in categories or []:
        srcs.extend(CATEGORY_SOURCES.get(c, []))
    if not srcs:
        srcs = CATEGORY_SOURCES["*"]

    # fetch items
    results = []
    seen = set()
    for url in srcs:
        for it in _fetch_rss(url):
            key = (it["title"], it["link"])
            if key in seen:
                continue
            seen.add(key)
            results.append(it)
        if len(results) >= max_items * 2:
            break
        time.sleep(0.25)

    # light keyword filter by merchants; if that empties list, fall back
    if merchants:
        kws = {w.lower() for w in merchants if len(w) >= 3}
        filtered = [r for r in results if any(k in r["title"].lower() for k in kws)]
        if filtered:
            results = filtered

    # guaranteed non-empty fallback
    if not results:
        results = [
            {"title": "Cut recurring subscriptions you don’t use", "link": "https://www.reddit.com/r/frugal/", "source": "guide"},
            {"title": "Groceries: meal plan + batch cook to save 20–30%", "link": "https://www.reddit.com/r/EatCheapAndHealthy/", "source": "guide"},
            {"title": "Switch to a lower-cost mobile plan (MVNO)", "link": "https://www.reddit.com/r/frugal/", "source": "guide"},
        ]

    return results[:max_items]
