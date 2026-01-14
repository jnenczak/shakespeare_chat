import json
import re
import random
from pathlib import Path

# --- ALWAYS derive paths from this script location ---
SCRIPT_DIR = Path(__file__).resolve().parent          # .../scripts
PROJECT_ROOT = SCRIPT_DIR.parent                      # .../shakespeare_tinyllama
DATA_DIR = PROJECT_ROOT / "data"

INSTRUCTIONS = [
    "Write a short monologue about {topic}.",
    "Explain {topic} in the style of William Shakespeare.",
    "Compose a brief speech on {topic}, in Shakespearean diction.",
    "Write a dramatic passage concerning {topic}.",
]

TOPICS = [
    "love", "jealousy", "ambition", "betrayal", "honor", "revenge",
    "time", "death", "fate", "guilt", "madness", "power"
]

def strip_gutenberg_header_footer(text: str) -> str:
    start = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*", text, re.IGNORECASE)
    end   = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*\*\*\*", text, re.IGNORECASE)
    if start and end and start.end() < end.start():
        return text[start.end():end.start()]
    return text

def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_by_paragraphs(text: str, min_chars=600, max_chars=1600):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(p) < 30:
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if len(buf) >= min_chars:
                chunks.append(buf)
            buf = p
    if len(buf) >= min_chars:
        chunks.append(buf)
    return chunks

def main():
    in_path = DATA_DIR / "shakespeare.txt"
    out_path = DATA_DIR / "sft_style.jsonl"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing file: {in_path}")

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    text = strip_gutenberg_header_footer(text)
    text = normalize(text)

    chunks = chunk_by_paragraphs(text)
    random.shuffle(chunks)

    # Start small; increase later if needed
    n = min(4000, len(chunks))

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            topic = random.choice(TOPICS)
            instr = random.choice(INSTRUCTIONS).format(topic=topic)

            sample = {
                "text": (
                    "### System:\n"
                    "You write in a Shakespearean style: Elizabethan diction, metaphor, dramatic cadence. "
                    "Avoid modern slang.\n\n"
                    f"### Instruction:\n{instr}\n\n"
                    f"### Response:\n{chunks[i]}\n"
                )
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Saved {n} examples to {out_path}")

if __name__ == "__main__":
    main()
