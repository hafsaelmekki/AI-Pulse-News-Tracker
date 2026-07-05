from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    azure_key = os.getenv("AZURE_AI_KEY")
    news_key = os.getenv("NEWS_API_KEY")

    print("--- Configuration check ---")
    print("Azure AI key:", "found" if azure_key else "missing")
    print("NewsAPI key:", "found" if news_key else "missing")

    if not news_key:
        return 1

    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={"q": "AI", "pageSize": 1, "apiKey": news_key},
        timeout=20,
    )
    if response.ok:
        print("NewsAPI connectivity: ok")
        return 0

    print(f"NewsAPI connectivity: failed ({response.status_code})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
