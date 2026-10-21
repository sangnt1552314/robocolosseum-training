"""Copy LingBot-VLA's TensorBoard scalars into its W&B run.

LingBot-VLA writes its training scalars (loss, LR, grad norm, MoE stats, ...) only to TensorBoard
(<output_dir>/runs); its W&B run holds just the config and system metrics (the same on Hopper). After
training, this script reads the event files and logs every scalar to the same W&B run, resumed by id,
against a `global_step` x-axis, so the curves show up next to the other models in RoboColosseum.

Idempotent: the run summary records the last copied step (tb_backfill/last_step); a rerun skips those.
Run it only when the training job has finished (two writers on one W&B run conflict).

Usage (with the LingBot venv, after sourcing scripts/nscc/env.sh):
  python scripts/lingbot/sync_tb_to_wandb.py --logdir <output_dir>/runs --run-id <job name> [--dry-run]
"""

import argparse
import glob
import os
from collections import defaultdict


def read_scalars(logdir: str) -> dict[int, dict[str, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows: dict[int, dict[str, float]] = defaultdict(dict)
    # Event files are named by creation time, so a later (resubmitted) run wins on a step clash.
    for path in sorted(glob.glob(os.path.join(logdir, "**", "events.out.tfevents.*"), recursive=True)):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for event in acc.Scalars(tag):
                rows[event.step][tag] = event.value
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "RoboColosseum"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--dry-run", action="store_true", help="only report what would be logged")
    args = parser.parse_args()

    rows = read_scalars(args.logdir)
    if not rows:
        raise SystemExit(f"No TensorBoard scalars under {args.logdir}")
    tags = sorted({tag for values in rows.values() for tag in values})
    steps = sorted(rows)
    print(f"{len(steps)} steps ({steps[0]}..{steps[-1]}), {len(tags)} tags: {tags[:12]}{' ...' if len(tags) > 12 else ''}")
    if args.dry_run:
        return

    import wandb

    run = wandb.init(id=args.run_id, project=args.project, entity=args.entity, resume="allow")
    last = run.summary.get("tb_backfill/last_step", -1)
    run.define_metric("global_step")
    for tag in tags:
        run.define_metric(tag, step_metric="global_step")
    logged = 0
    for step in steps:
        if step <= last:
            continue
        run.log({"global_step": step, **rows[step]})
        logged += 1
    run.summary["tb_backfill/last_step"] = steps[-1]
    run.finish()
    print(f"Logged {logged} new steps to {args.entity}/{args.project}/{args.run_id} (skipped {len(steps) - logged}).")


if __name__ == "__main__":
    main()
