"""Build g1-carry-box-left-eye from g1-carry-box (LeRobot v3), leaving the source untouched.

- observation.images.head: cropped to the left eye (480x1280 stereo -> 480x640) with
  tools/g1_head_left_eye.py, so all three cameras share one shape (LeRobot's GR00T processor
  stacks cameras before resizing and needs equal shapes).
- All cameras are re-encoded from AV1 to H.264 (libx264, yuv420p, keyframe every 2 frames)
  for faster decoding during training.
- Every video file is re-encoded frame by frame with its original timestamps and file layout,
  so meta/episodes (chunk/file index, from/to timestamps) stays valid unchanged. Frame counts
  and timestamps are verified against the source for every file.
- data/ parquet, tasks, calibration and the rest of meta/ are copied as-is; info.json gets the
  new head shape and codec, and a note is appended to README.md.

Usage: python tools/make_g1_left_eye_dataset.py <src_root> <dst_root> [--workers N]
"""

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import av

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_head_left_eye import EYE_WIDTH, crop_head_left_eye  # noqa: E402

HEAD_KEY = "observation.images.head"
CODEC = "libx264"
GOP = 2
CRF = 23


def reencode(src: Path, dst: Path, crop: bool) -> tuple[str, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_pts = []
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        in_stream = inp.streams.video[0]
        in_stream.codec_context.thread_count = 2
        width = EYE_WIDTH if crop else in_stream.codec_context.width
        out_stream = out.add_stream(CODEC, rate=in_stream.average_rate)
        out_stream.width = width
        out_stream.height = in_stream.codec_context.height
        out_stream.pix_fmt = "yuv420p"
        out_stream.time_base = in_stream.time_base
        out_stream.codec_context.time_base = in_stream.time_base
        out_stream.options = {"g": str(GOP), "crf": str(CRF), "threads": "2"}
        for frame in inp.decode(in_stream):
            arr = frame.to_ndarray(format="rgb24")
            if crop:
                arr = crop_head_left_eye(arr)
            new = av.VideoFrame.from_ndarray(arr, format="rgb24")
            new.pts = frame.pts
            new.time_base = in_stream.time_base
            src_pts.append(frame.pts)
            for packet in out_stream.encode(new):
                out.mux(packet)
        for packet in out_stream.encode():
            out.mux(packet)

    with av.open(str(dst)) as chk:
        stream = chk.streams.video[0]
        out_pts = [f.pts for f in chk.decode(stream)]
        if (stream.codec_context.width, stream.codec_context.height) != (width, out_stream.height):
            raise RuntimeError(f"{dst}: unexpected size {stream.codec_context.width}x{stream.codec_context.height}")
    if [p * stream.time_base for p in out_pts] != [p * in_stream.time_base for p in src_pts]:
        raise RuntimeError(f"{dst}: frame timestamps differ from {src} ({len(out_pts)} vs {len(src_pts)} frames)")
    return str(dst), len(src_pts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if dst.exists():
        sys.exit(f"{dst} already exists; remove it first.")
    info = json.loads((src / "meta/info.json").read_text())
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    assert info["features"][HEAD_KEY]["shape"][1] == 2 * EYE_WIDTH, "head is not a side-by-side stereo frame"

    # Build in a temp dir and rename at the end, so a failed run never leaves a half dataset.
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("videos"))

    jobs = []
    for key in video_keys:
        for f in sorted((src / "videos" / key).glob("*/*.mp4")):
            jobs.append((f, tmp / f.relative_to(src), key == HEAD_KEY))
    print(f"Re-encoding {len(jobs)} video files with {args.workers} workers", flush=True)

    total = {k: 0 for k in video_keys}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(reencode, *job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            path, n = fut.result()
            key = Path(path).relative_to(tmp / "videos").parts[0]
            total[key] += n
            print(f"[{i}/{len(jobs)}] {Path(path).relative_to(tmp)} ({n} frames)", flush=True)
    for key, n in total.items():
        if n != info["total_frames"]:
            raise RuntimeError(f"{key}: {n} frames re-encoded, expected {info['total_frames']}")

    for key in video_keys:
        feat = info["features"][key]
        if key == HEAD_KEY:
            feat["shape"][1] = EYE_WIDTH
            feat["info"]["video.width"] = EYE_WIDTH
        feat["info"]["video.codec"] = "h264"
        feat["info"]["video.pix_fmt"] = "yuv420p"
    (tmp / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")

    with open(tmp / "README.md", "a") as f:
        f.write(
            "\n## Left-eye variant\n\n"
            f"Derived from `{src.name}` by tools/make_g1_left_eye_dataset.py "
            "(robocolosseum-training). `observation.images.head` keeps only the left eye of the "
            "stereo frame (columns 0-639 of the 480 x 1280 side-by-side image), so it is 480 x 640 "
            f"like the wrist cameras. All videos are re-encoded to H.264 ({CODEC}, crf {CRF}, "
            f"keyframe every {GOP} frames) with the original timestamps and file layout; numeric "
            "data, episodes and language are unchanged. Image statistics in `meta/stats.json` and "
            "`meta/episodes` still describe the full stereo head frame.\n\n"
            "At inference, crop the robot's head frame the same way (tools/g1_head_left_eye.py, "
            "`crop_head_left_eye`) before passing it as `observation.images.head`.\n"
        )

    tmp.rename(dst)
    print(f"Done: {dst}")


if __name__ == "__main__":
    main()
