#!/usr/bin/env python3
"""Build slime prompt-data (jsonl) from a parquet of SWE instances.

slime drives GRPO over this file: for each line it calls the custom generate
(examples.cc_swe_rl.client.generate), which reads metadata.instance_id and
asks the cc-run service to roll out that instance. The actual task content
(problem_statement, FAIL_TO_PASS, image, ...) lives in cc-run's own dataset
(CC_RUN_RL_DATASET_PATH), keyed by the same instance_id — so this file only
needs to carry instance_id. A `prompt` field is included for slime's
--input-key but is not used to drive the agent.

Usage:
  python make_prompt_data.py --parquet <in.parquet> --out <train.jsonl> [--limit N]

The parquet and cc-run's dataset should be the SAME file (or share instance_ids).
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--instance-col", default="instance_id")
    ap.add_argument("--prompt-col", default="problem_statement",
                    help="column used only to populate the slime `prompt` field")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.limit:
        df = df.head(args.limit)

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            iid = row.get(args.instance_col)
            if iid is None or (isinstance(iid, float) and pd.isna(iid)):
                continue
            ps = row.get(args.prompt_col, "")
            if isinstance(ps, float) and pd.isna(ps):
                ps = ""
            rec = {
                # slime --input-key. Must be a message list (not a raw str):
                # when the hf checkpoint ships an AutoProcessor (Qwen VL family),
                # slime's Dataset asserts isinstance(prompt, list). Content is
                # irrelevant — the agent is driven by cc-run via instance_id.
                "prompt": [{"role": "user", "content": str(ps)[:2000]}],
                "label": str(iid),                 # convenience
                "metadata": {"instance_id": str(iid)},  # client reads this
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} prompts -> {args.out}")


if __name__ == "__main__":
    main()
