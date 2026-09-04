"""Split a composite pool into N symlinked shard pools so a flaky-network upload survives drops.

Each shard is a real MatteDataset dir (input/ + alpha/ symlinks + its own manifest.jsonl)
pointing back at the parent files — near-zero disk. Upload each shard independently; a drop
costs one ~900MB shard, not the whole 7G. The harness reads them all via --pools.

Run:
    python -m knockout.shard_composites --shards 8
    python -m knockout.shard_composites --src <lab>/data/v13_composites --data-dir <lab>/data --out-prefix v13_comp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
SRC = DATA / "v12_composites"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--src", type=Path, default=SRC, help="composite pool to shard")
    ap.add_argument("--data-dir", type=Path, default=DATA, help="where the shard dirs are written")
    ap.add_argument("--out-prefix", default="v12_comp",
                    help="shard dir name prefix; must not collide with pools already on the volume")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.src / "manifest.jsonl").open()]
    per = -(-len(rows) // args.shards)  # ceil
    made = []
    for s in range(args.shards):
        chunk = rows[s * per:(s + 1) * per]
        if not chunk:
            break
        d = args.data_dir / f"{args.out_prefix}_{s:02d}"
        (d / "input").mkdir(parents=True, exist_ok=True)
        (d / "alpha").mkdir(parents=True, exist_ok=True)
        mf = (d / "manifest.jsonl").open("w")
        for it in chunk:
            for sub in ("input", "alpha"):
                link = d / it[sub]
                if not link.exists():
                    link.symlink_to((args.src / it[sub]).resolve())
            mf.write(json.dumps(it) + "\n")
        mf.close()
        made.append((d.name, len(chunk)))
    print("shards:", ", ".join(f"{n}={c}" for n, c in made))
    print("pools arg: " + ",".join(n for n, _ in made) + ",teacher_train*6")


if __name__ == "__main__":
    main()
