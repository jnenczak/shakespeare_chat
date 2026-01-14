import json
import re
from pathlib import Path

import requests
import mwparserfromhell

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

WIKI_API = "https://en.wikipedia.org/w/api.php"

def fetch_wikitext(title: str) -> tuple[str, dict]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "prop": "revisions",
        "titles": title,
        "rvslots": "*",
        "rvprop": "content",
    }
    r = requests.get(WIKI_API, params=params, headers={"User-Agent": "local-shakespeare-rag/1.0"})
    r.raise_for_status()
    j = r.json()

    page = j["query"]["pages"][0]
    if "missing" in page:
        raise RuntimeError(f"Page missing for title: {title}")

    wikitext = page["revisions"][0]["slots"]["main"]["content"]
    meta = {
        "title": page.get("title"),
        "pageid": page.get("pageid"),
        "source_api": WIKI_API,
        "source_page": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "license_note": "Wikipedia text is CC BY-SA; keep attribution if you redistribute.",
    }
    return wikitext, meta

def wikitext_to_plaintext(wikitext: str) -> str:
    code = mwparserfromhell.parse(wikitext)
    text = code.strip_code(normalize=True, collapse=True)

    # remove citation markers like [1], [23]
    text = re.sub(r"\[\d+\]", "", text)

    # normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def main():
    title = "William Shakespeare"

    wikitext, meta = fetch_wikitext(title)

    (DATA_DIR / "wiki_shakespeare_full.wiki").write_text(wikitext, encoding="utf-8")
    plain = wikitext_to_plaintext(wikitext)
    (DATA_DIR / "wiki_shakespeare_full.txt").write_text(plain, encoding="utf-8")
    (DATA_DIR / "wiki_shakespeare_full_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved wikitext: ", DATA_DIR / "wiki_shakespeare_full.wiki")
    print("Saved plaintext:", DATA_DIR / "wiki_shakespeare_full.txt")
    print("Saved meta:     ", DATA_DIR / "wiki_shakespeare_full_meta.json")

if __name__ == "__main__":
    main()
