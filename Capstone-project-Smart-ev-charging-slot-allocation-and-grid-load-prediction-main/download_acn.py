from urllib.parse import urljoin

import pandas as pd
import requests

API_KEY = "OX0WnYJVUbroXl0TK0MQEY7raD0fHGnLgBl-Ez-Nuos"
BASE_URL = "https://ev.caltech.edu/api/v1/"
START_ENDPOINT = "sessions/caltech?max_results=100&page=1"
TARGET_ROWS = 600

headers = {"Authorization": f"Bearer {API_KEY}"}

all_items = []
next_href = START_ENDPOINT

with requests.Session() as session:
    session.headers.update(headers)

    while next_href and len(all_items) < TARGET_ROWS:
        page_url = (
            next_href if next_href.startswith("http") else urljoin(BASE_URL, next_href)
        )
        response = session.get(page_url, timeout=30)
        response.raise_for_status()
        data = response.json()

        page_items = data.get("_items", [])
        if not page_items:
            break

        all_items.extend(page_items)
        next_href = data.get("_links", {}).get("next", {}).get("href")

df = pd.json_normalize(all_items[:TARGET_ROWS])
df.to_csv("acn_data.csv", index=False)

print("Saved", len(df), "rows")
