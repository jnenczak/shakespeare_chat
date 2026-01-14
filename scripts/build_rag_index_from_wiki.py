import argparse
import os
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"


def set_cwd_to_script_dir():
    os.chdir(SCRIPT_DIR)


def clean_text_basic(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_by_paragraphs(text: str, max_chars: int = 1200, min_chars: int = 250):
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks = []
    cur = []
    cur_len = 0

    for p in paras:
        if len(p) < 30:
            continue

        if cur_len + len(p) + 2 > max_chars and cur_len >= min_chars:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p) + 2

    if cur and cur_len >= min_chars:
        chunks.append("\n\n".join(cur))

    return chunks


def main():
    set_cwd_to_script_dir()

    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki_txt", default=str(DATA_DIR / "wiki_shakespeare_full.txt"))
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk_max_chars", type=int, default=1200)
    ap.add_argument("--chunk_min_chars", type=int, default=250)

    ap.add_argument("--rag_dir", default=str(OUT_DIR / "rag_wiki_shakespeare"))
    args = ap.parse_args()

    wiki_path = Path(args.wiki_txt)
    if not wiki_path.exists():
        raise FileNotFoundError(f"Missing Wikipedia TXT: {wiki_path}")

    rag_dir = Path(args.rag_dir)
    rag_dir.mkdir(parents=True, exist_ok=True)
    index_path = rag_dir / "faiss.index"
    chunks_path = rag_dir / "chunks.txt"
    meta_path = rag_dir / "meta.txt"

    raw = wiki_path.read_text(encoding="utf-8", errors="ignore")
    raw = clean_text_basic(raw)
    chunks = chunk_by_paragraphs(raw, max_chars=args.chunk_max_chars, min_chars=args.chunk_min_chars)

    if not chunks:
        raise RuntimeError("No chunks created. Check your wiki txt formatting.")

    print(f"[RAG] Chunks: {len(chunks)}")
    print(f"[RAG] Embedding model: {args.embed_model}")

    embedder = SentenceTransformer(args.embed_model)
    emb = embedder.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  
    index.add(emb)

    faiss.write_index(index, str(index_path))
    chunks_path.write_text("\n\n---CHUNK---\n\n".join(chunks), encoding="utf-8")
    meta_path.write_text(
        f"embed_model={args.embed_model}\nchunks={len(chunks)}\n",
        encoding="utf-8",
    )

    print(f"[RAG] Saved index:  {index_path}")
    print(f"[RAG] Saved chunks: {chunks_path}")
    print(f"[RAG] Saved meta:   {meta_path}")


if __name__ == "__main__":
    main()
