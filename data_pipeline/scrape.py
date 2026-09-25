import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

BASE = "https://books.toscrape.com/"
OUTPUT = Path(__file__).parent / "books_clean.csv"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

def get_soup(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def scrape_category(category_url, category_name):
    rows = []
    url = category_url
    while url:
        soup = get_soup(url)
        for article in soup.select("article.product_pod"):
            title = article.h3.a.get("title", "").strip()
            price_text = article.select_one(".price_color").get_text(strip=True)
            rating_text = next(
                (c for c in article.select_one(".star-rating").get("class", [])
                 if c in RATING_MAP), None
            )
            availability = article.select_one(".availability").get_text(" ", strip=True)
            rows.append({
                "title": title,
                "price": price_text,
                "star_rating": rating_text,
                "availability": availability,
                "category": category_name,
            })
        nxt = soup.select_one("li.next a")
        if nxt:
            from urllib.parse import urljoin
            url = urljoin(url, nxt["href"])
        else:
            url = None
    return rows

def main():
    soup = get_soup(BASE)
    links = soup.select(".side_categories ul li ul li a")
    selected = links[:3]
    rows = []
    for a in selected:
        from urllib.parse import urljoin
        rows.extend(scrape_category(urljoin(BASE, a["href"]), a.get_text(strip=True)))

    df = pd.DataFrame(rows).drop_duplicates(subset=["title", "category"])

    df["price_gbp"] = pd.to_numeric(
        df["price"].str.replace("£", "", regex=False), errors="coerce"
    )
    df["rating"] = df["star_rating"].map(RATING_MAP)
    df["in_stock"] = df["availability"].str.contains("In stock", case=False, na=False)

    # Required baseline: 1 GBP = 105.50 INR.
    df["price_inr"] = df["price_gbp"] * 105.50

    # Defensive handling for unexpected parse failures.
    df = df.dropna(subset=["title", "price_gbp", "rating"]).copy()
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} books to {OUTPUT}")
    print(df.head())

if __name__ == "__main__":
    main()
