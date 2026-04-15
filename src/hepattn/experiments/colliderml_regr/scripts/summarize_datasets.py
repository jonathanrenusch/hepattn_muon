"""Report total tracks per preprocessed dataset under a given root.

Reads each dataset's ``manifest.json`` and prints / saves a summary table with
event and track counts, plus the selection used to produce it.

Usage:
    pixi run python -m hepattn.experiments.colliderml_regr.scripts.summarize_datasets \\
        --root /eos/project/e/end-to-end-colliderml/data/NeurIPS_retraining \\
        [--output summary.txt]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _count_one_shard(shard_dir: Path) -> tuple[int, int, int, int]:
    """Return (n_tracks, n_events, n_hits, n_dm) for a single shard.

    Reads small .npy headers without loading full arrays where possible.
    """
    sel = shard_dir / "selected_tracks"
    offsets_file = sel / "track_hit_offsets.npy"
    event_idx_file = sel / "track_event_idx.npy"
    dm_file = sel / "acts_dm_mask.npy"

    if not offsets_file.exists():
        return (0, 0, 0, 0)

    offsets = np.load(offsets_file, mmap_mode="r")
    n_tracks = max(len(offsets) - 1, 0)
    n_hits = int(offsets[-1]) if n_tracks > 0 else 0

    n_events = 0
    if event_idx_file.exists():
        ev = np.load(event_idx_file, mmap_mode="r")
        n_events = int(ev.max()) + 1 if len(ev) > 0 else 0

    n_dm = 0
    if dm_file.exists():
        dm = np.load(dm_file, mmap_mode="r")
        n_dm = int(np.count_nonzero(dm))

    return (n_tracks, n_events, n_hits, n_dm)


def count_shards(dataset_dir: Path, max_workers: int = 16) -> dict:
    shard_dirs = sorted(dataset_dir.glob("shard_*"))
    if not shard_dirs:
        return {"n_tracks": 0, "n_events": 0, "n_hits": 0, "n_dm": 0, "n_shards": 0}

    totals = {"n_tracks": 0, "n_events": 0, "n_hits": 0, "n_dm": 0}
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_count_one_shard, d): d for d in shard_dirs}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"Counting {dataset_dir.name}", file=sys.stderr):
            nt, ne, nh, nd = fut.result()
            totals["n_tracks"] += nt
            totals["n_events"] += ne
            totals["n_hits"] += nh
            totals["n_dm"] += nd
    totals["n_shards"] = len(shard_dirs)
    return totals


def summarize(root: Path, recount: bool = False, max_workers: int = 16) -> list[dict]:
    rows = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        dataset = manifest_path.parent.name
        try:
            with open(manifest_path) as f:
                m = json.load(f)
        except Exception as exc:
            rows.append({"dataset": dataset, "error": str(exc)})
            continue
        totals = m.get("totals", {})
        row = {
            "dataset": dataset,
            "n_events": totals.get("n_events"),
            "n_tracks": m.get("total_tracks", totals.get("n_selected_tracks")),
            "n_hits": totals.get("n_selected_hits"),
            "n_acts_dm": totals.get("n_acts_double_matched"),
            "selection": m.get("selection", {}),
            "source": "manifest",
        }
        if recount or not row["n_tracks"]:
            counted = count_shards(manifest_path.parent, max_workers=max_workers)
            row.update({
                "n_events": counted["n_events"],
                "n_tracks": counted["n_tracks"],
                "n_hits": counted["n_hits"],
                "n_acts_dm": counted["n_dm"],
                "n_shards": counted["n_shards"],
                "source": "shard-scan",
            })
        rows.append(row)
    return rows


def fmt_int(n) -> str:
    return f"{n:,}" if isinstance(n, int) else "—"


def render(rows: list[dict]) -> str:
    lines: list[str] = []
    header = f"{'Dataset':<35} {'Events':>12} {'Tracks':>16} {'Hits':>18} {'ACTS DM tracks':>16}  {'Source':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    total_events = 0
    total_tracks = 0
    total_hits = 0
    total_dm = 0
    for r in rows:
        if "error" in r:
            lines.append(f"{r['dataset']:<35} ERROR: {r['error']}")
            continue
        lines.append(
            f"{r['dataset']:<35} {fmt_int(r['n_events']):>12} "
            f"{fmt_int(r['n_tracks']):>16} {fmt_int(r['n_hits']):>18} "
            f"{fmt_int(r['n_acts_dm']):>16}  {r.get('source', 'manifest'):>10}"
        )
        total_events += r["n_events"] or 0
        total_tracks += r["n_tracks"] or 0
        total_hits += r["n_hits"] or 0
        total_dm += r["n_acts_dm"] or 0
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<35} {total_events:>12,} {total_tracks:>16,} "
        f"{total_hits:>18,} {total_dm:>16,}"
    )
    lines.append("")
    lines.append("Per-dataset selection used at preprocessing:")
    for r in rows:
        if "error" in r:
            continue
        sel = r["selection"]
        key_fields = [
            ("min_hits", sel.get("min_hits")),
            ("max_hits", sel.get("max_hits")),
            ("primary", sel.get("primary")),
            ("hard_scatter", sel.get("hard_scatter")),
            ("charged", sel.get("charged")),
            ("pt_min", sel.get("pt_min")),
            ("eta", f"[{sel.get('eta_min')},{sel.get('eta_max')}]"),
            ("d0", f"[{sel.get('d0_min')},{sel.get('d0_max')}]"),
            ("z0", f"[{sel.get('z0_min')},{sel.get('z0_max')}]"),
            ("require_acts_dm", sel.get("require_acts_dm")),
        ]
        pretty = ", ".join(f"{k}={v}" for k, v in key_fields if v is not None)
        lines.append(f"  {r['dataset']}: {pretty}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--recount", action="store_true",
                    help="Always scan shards even when manifest reports non-zero counts")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    rows = summarize(Path(args.root), recount=args.recount, max_workers=args.workers)
    if not rows:
        print(f"No manifest.json found under {args.root}")
        return
    text = render(rows)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
