"""Backfill model-baseline descriptions that the original homepage scrape missed,
fetching the page through Firecrawl instead. No vendor (model) list calls.

For every candidate whose description_source is not "scraped_homepage" (site
blocked us, timed out, DNS failure, JavaScript-only page, ...), this:

  1. Fetches https://<domain> once per unique domain through Firecrawl's
     /v1/scrape endpoint (shared across the vendors being backfilled).
  2. If Firecrawl returns the page (HTTP < 400) with at least 80 characters of
     text, the SAME model that proposed the candidate writes the description
     from that text only, with the runner's own DESC_SYSTEM prompt (no outside
     knowledge). Opus candidates are described by Opus, Sol's by Sol.
  3. If Firecrawl also fails, the model is NOT called and the description stays
     empty. Nothing is generated from memory.

It updates `extracted_candidates` in each cell's raw audit file (and appends the
Firecrawl fetch to `vendor_calls`, auth never stored). Re-judge afterwards with
scripts/rejudge_lookalike_from_raw.py, which reads those candidates.

    env -u FIRECRAWL_API_KEY PYTHONPATH=scripts .venv/bin/python \\
        scripts/backfill_descriptions_firecrawl.py --only claude-opus-5-5,gpt-5-6-sol

--dry-run fetches and describes but writes only the report, not the raw files.
--limit N backfills only the first N missing candidates (for a test run).
Firecrawl results are cached in --cache so an interrupted run resumes cheaply.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lookalike.common import load_dotenv, now_iso, read_snapshot
from lookalike.runners import model_baseline as base
from lookalike.runners import model_baseline_sol as sol

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "lookalike-runs"
FIRECRAWL_URL = "https://api.firecrawl.dev/v1/scrape"
FIRECRAWL_TIMEOUT_MS = 45000
MIN_TEXT_CHARS = 80  # same threshold as the original scrape

DESCRIBERS = {
    "claude-opus-5-5": base._converse,
    "gpt-5-6-sol": sol._complete,
}

_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WS = re.compile(r"[ \t]+")


def markdown_to_text(md: str) -> str:
    """Drop images and link targets so the text budget goes to words."""
    md = _MD_IMAGE.sub("", md)
    md = _MD_LINK.sub(r"\1", md)
    lines = [_WS.sub(" ", ln).strip() for ln in md.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def firecrawl_fetch(domain: str, key: str) -> dict[str, Any]:
    """One Firecrawl scrape of the homepage. Never raises."""
    body = {
        "url": f"https://{domain}",
        "formats": ["markdown"],
        "onlyMainContent": True,
        "timeout": FIRECRAWL_TIMEOUT_MS,
    }
    started = time.monotonic()
    last = "no attempt"
    for attempt in range(4):
        req = urllib.request.Request(
            FIRECRAWL_URL,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=FIRECRAWL_TIMEOUT_MS / 1000 + 30) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            last = f"firecrawl HTTP {exc.code}: {detail}"
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 2)
                continue
            break
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
            continue
        data = payload.get("data") or {}
        meta = data.get("metadata") or {}
        status = meta.get("statusCode")
        text = markdown_to_text(data.get("markdown") or "")[: base.SCRAPE_TEXT_CHARS]
        ok = bool(payload.get("success")) and (status is None or int(status) < 400)
        return {
            "ok": ok and len(text) >= MIN_TEXT_CHARS,
            "status": status,
            "final_url": meta.get("url") or meta.get("sourceURL") or body["url"],
            "text": text if ok else "",
            "error": None if ok else (payload.get("error") or f"page status {status}"),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }
    return {"ok": False, "status": None, "final_url": body["url"], "text": "",
            "error": last[:300], "elapsed_ms": int((time.monotonic() - started) * 1000)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="claude-opus-5-5,gpt-5-6-sol")
    parser.add_argument("--seeds", help="comma-separated seed slugs (default: all)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fetch-concurrency", type=int, default=50)
    parser.add_argument("--describe-concurrency", type=int, default=12)
    parser.add_argument("--cache", default=str(ROOT / "data" / "firecrawl-cache.json"))
    parser.add_argument("--report", default=str(ROOT / "data" / "firecrawl-backfill-report.json"))
    args = parser.parse_args()

    load_dotenv()
    key = os.environ.get("FIRECRAWL_API_KEY")
    if not key:
        print("FIRECRAWL_API_KEY missing", file=sys.stderr)
        return 1
    snapshot = read_snapshot()
    raw_dir = RUNS_DIR / snapshot["dataset_slug"]
    vendors = [v.strip() for v in args.only.split(",") if v.strip()]
    for v in vendors:
        if v not in DESCRIBERS:
            print(f"no describer for {v}", file=sys.stderr)
            return 1
    seeds = [s["seed_slug"] for s in snapshot["seeds"]]
    if args.seeds:
        want = set(args.seeds.split(","))
        seeds = [s for s in seeds if s in want]

    # Work list: (raw path, candidate index) for every missing description.
    raws: dict[Path, dict[str, Any]] = {}
    todo: list[tuple[Path, str, int]] = []
    for s in seeds:
        for v in vendors:
            p = raw_dir / s / f"{v}.raw.json"
            if not p.is_file():
                continue
            raw = json.loads(p.read_text(encoding="utf-8"))
            cands = raw["attempts"][0]["extracted_candidates"] or []
            for i, c in enumerate(cands):
                if (c.get("extra") or {}).get("description_source") != "scraped_homepage" and c.get("domain"):
                    todo.append((p, v, i))
            raws[p] = raw
    if args.limit:
        todo = todo[: args.limit]
    domains = sorted({raws[p]["attempts"][0]["extracted_candidates"][i]["domain"].strip().lower() for p, _, i in todo})
    print(f"missing descriptions: {len(todo)} across {len({p for p, _, _ in todo})} cells; unique domains: {len(domains)}")

    cache_path = Path(args.cache)
    cache: dict[str, Any] = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
    lock = threading.Lock()
    need = [d for d in domains if d not in cache]
    print(f"firecrawl: {len(need)} to fetch ({len(domains) - len(need)} cached)")
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.fetch_concurrency) as ex:
        futs = {ex.submit(firecrawl_fetch, d, key): d for d in need}
        for fut in concurrent.futures.as_completed(futs):
            d = futs[fut]
            with lock:
                cache[d] = fut.result()
                done += 1
                if done % 100 == 0 or done == len(need):
                    cache_path.write_text(json.dumps(cache))
                    print(f"  fetched {done}/{len(need)}", flush=True)
    cache_path.write_text(json.dumps(cache))

    usage: dict[str, dict[str, int]] = {v: {} for v in vendors}
    outcome: dict[str, dict[str, int]] = {v: {} for v in vendors}

    def describe(item: tuple[Path, str, int]) -> None:
        p, v, i = item
        c = raws[p]["attempts"][0]["extracted_candidates"][i]
        extra = c.setdefault("extra", {})
        domain = c["domain"].strip().lower()
        fc = cache[domain]
        prior = extra.get("description_source")
        extra.setdefault("original_description_source", prior)
        if extra.get("scrape_error"):
            extra.setdefault("original_scrape_error", extra.pop("scrape_error"))
        extra.pop("describe_error", None)
        if not fc["ok"]:
            # Firecrawl failed too: no model call, description stays empty.
            c["description"] = None
            extra["description_source"] = "firecrawl_failed"
            extra["firecrawl_error"] = (fc.get("error") or ("too little text" if fc.get("text") is not None else ""))[:200]
            key_ = f"{prior} -> firecrawl_failed"
        else:
            try:
                desc = DESCRIBERS[v](
                    base.DESC_SYSTEM,
                    f"Company: {c['name']}\nDomain: {domain}\n\nWebsite text:\n{fc['text']}",
                    base.DESC_MAX_TOKENS,
                    usage[v],
                )
                c["description"] = desc.strip()[:1000]
                extra["description_source"] = "firecrawl_homepage"
                extra.pop("firecrawl_error", None)
                key_ = f"{prior} -> firecrawl_homepage"
            except Exception as exc:  # noqa: BLE001
                c["description"] = None
                extra["description_source"] = "describe_failed"
                extra["describe_error"] = str(exc)[:200]
                key_ = f"{prior} -> describe_failed"
        with lock:
            outcome[v][key_] = outcome[v].get(key_, 0) + 1
            raws[p]["attempts"][0]["vendor_calls"].append({
                "method": "POST",
                "url": FIRECRAWL_URL,
                "request_headers": {"Authorization": "***REDACTED***"},
                "request_body": {"url": f"https://{domain}", "formats": ["markdown"]},
                "response_status": fc.get("status") or 0,
                "response_body": {"final_url": fc.get("final_url"), "text_chars": len(fc.get("text") or ""),
                                  "text_preview": (fc.get("text") or "")[:1500], "error": fc.get("error")},
                "elapsed_ms": fc.get("elapsed_ms") or 0,
            })

    print(f"describing (only where Firecrawl returned text) ...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.describe_concurrency) as ex:
        list(ex.map(describe, todo))

    report = {
        "at": now_iso(), "dry_run": args.dry_run, "candidates": len(todo),
        "unique_domains": len(domains), "outcome": outcome, "describe_usage": usage,
        "samples": [
            {k: raws[p]["attempts"][0]["extracted_candidates"][i].get(k) for k in ("name", "domain", "description", "extra")}
            | {"vendor": v, "seed": p.parent.name}
            for p, v, i in todo[:40]
        ],
    }
    Path(args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"outcome": outcome, "describe_usage": usage}, indent=1))
    if not args.dry_run:
        for p in {p for p, _, _ in todo}:
            p.write_text(json.dumps(raws[p], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated {len({p for p, _, _ in todo})} raw files")
    else:
        print("--dry-run: raw files not modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
