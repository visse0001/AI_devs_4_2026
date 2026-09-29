import csv
import io
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CENTRAL_AI_DEVS")
CSV_URL = f"https://hub.ag3nts.org/data/{API_KEY}/categorize.csv"
VERIFY_URL = "https://hub.ag3nts.org/verify"

# Statyczny początek (cache), zmienne dane na końcu. Po angielsku = mniej tokenów.
PROMPT_TEMPLATE = (
    "Classify item. Reply only DNG or NEU.\n"
    "DNG: weapons, explosives, toxic, flammable, radioactive, dangerous.\n"
    "Anything reactor-related (fuel cassettes, reactor parts) is always NEU.\n"
    "Else NEU.\n"
    "ID {code}: {description}"
)


def send(prompt: str) -> dict:
    payload = {"apikey": API_KEY, "task": "categorize", "answer": {"prompt": prompt}}
    resp = requests.post(VERIFY_URL, json=payload)
    try:
        return resp.json()
    except ValueError:
        return {"status": resp.status_code, "text": resp.text}


def fetch_items() -> list[dict[str, str]]:
    resp = requests.get(CSV_URL)
    resp.raise_for_status()
    return list(csv.DictReader(io.StringIO(resp.text)))


def count_tokens(text: str) -> int | None:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text))
    except ImportError:
        return None


def run() -> None:
    print("RESET:", send("reset"))

    items = fetch_items()  # zawsze świeży CSV
    for item in items:
        prompt = PROMPT_TEMPLATE.format(**item)
        tokens = count_tokens(prompt)
        if tokens and tokens > 100:
            print(f"UWAGA: {item['code']} ma {tokens} tokenów (>100)")

        result = send(prompt)
        print(f"{item['code']} | {item['description'][:60]} -> {result}")

        text = str(result)
        if "FLG" in text:
            print("\nFLAGA:", text)
            return
        if isinstance(result, dict) and result.get("code", 0) < 0:
            print("\nBłąd, przerywam. Popraw prompt i uruchom ponownie.")
            return


if __name__ == "__main__":
    run()
