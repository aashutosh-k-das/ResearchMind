"""Command-line helpers.

    python -m app.cli ingest data/papers      # bulk-ingest a folder of PDFs
    python -m app.cli ask "What is LoRA?"     # one-off question
    python -m app.cli list                    # show indexed papers
"""
import argparse
import sys
from pathlib import Path

from . import ingest, rag


def main() -> None:
    p = argparse.ArgumentParser(prog="researchmind")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest", help="ingest all PDFs in a folder (recursive)")
    pi.add_argument("folder", type=Path)
    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("-k", "--top-k", type=int, default=None)
    sub.add_parser("list", help="list indexed papers")
    args = p.parse_args()

    if args.cmd == "ingest":
        def progress(i, total, r):
            extra = f"{r.get('chunks', '')} chunks" if r["status"] == "ingested" else r.get("error") or r.get("reason", "")
            print(f"[{i}/{total}] {r['status']:8} {r['file_name']}  {extra}", flush=True)
        results = ingest.ingest_folder(args.folder, progress=progress)
        ok = sum(r["status"] == "ingested" for r in results)
        print(f"\nDone: {ok} ingested, {sum(r['status'] == 'skipped' for r in results)} skipped, "
              f"{sum(r['status'] == 'error' for r in results)} failed")
    elif args.cmd == "ask":
        for ev in rag.answer_stream(args.question, args.top_k):
            if ev["type"] == "token":
                sys.stdout.write(ev["text"])
                sys.stdout.flush()
            elif ev["type"] == "sources":
                sources = ev["sources"]
        print("\n\nSources:")
        for s in sources:
            print(f"  [{s['id']}] {s['title']} — p.{s['page']} ({s['file_name']}, score {s['score']})")
    elif args.cmd == "list":
        for d in ingest.list_documents():
            print(f"{d['doc_id']}  {d['pages']:>3}p {d['chunks']:>4}c  {d['title']}")


if __name__ == "__main__":
    main()
