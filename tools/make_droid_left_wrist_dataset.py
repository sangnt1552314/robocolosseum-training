"""Build a left + wrist copy of a DROID LeRobot v3 dataset, leaving the source untouched.

Used by the pi05 and GR00T runs, which train on the left and wrist cameras only:
- observation.images.right_image is removed with LeRobot's own remove_feature (the same
  operation as `lerobot-edit-dataset --operation.type remove_feature`); the remaining videos,
  data parquet and meta are copied without re-encoding.
- The "features" dict in meta/info.json is reordered so observation.images.left_image comes
  before observation.images.wrist. LeRobot's GR00T processor (with relative actions) and pi05
  (without an input_features override) take their camera order from info.json, so this makes
  the model camera order left, wrist. Video keys are looked up by name, never by position, so
  the reorder is safe.

The copy is written to <dst_root>.tmp and renamed at the end, so an interrupted build never
leaves a half-written dataset at <dst_root>.

Usage: python tools/make_droid_left_wrist_dataset.py <src_root> <dst_root>
"""

import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets.dataset_tools import remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DROP_KEY = "observation.images.right_image"
CAMERA_ORDER = ["observation.images.left_image", "observation.images.wrist"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src_root", type=Path)
    parser.add_argument("dst_root", type=Path)
    args = parser.parse_args()

    src_root, dst_root = args.src_root.resolve(), args.dst_root.resolve()
    if (dst_root / "meta" / "info.json").exists():
        print(f"{dst_root} already exists; nothing to do.")
        return
    tmp_root = dst_root.with_name(dst_root.name + ".tmp")
    shutil.rmtree(tmp_root, ignore_errors=True)

    src = LeRobotDataset(f"local/{src_root.name}", root=src_root)
    remove_feature(src, feature_names=[DROP_KEY], output_dir=tmp_root, repo_id=f"local/{dst_root.name}")

    info_path = tmp_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = info["features"]
    assert DROP_KEY not in features, f"{DROP_KEY} still present"
    cameras = [k for k in features if features[k]["dtype"] == "video"]
    assert sorted(cameras) == sorted(CAMERA_ORDER), f"unexpected cameras: {cameras}"
    # Keep every non-camera feature where it was; put the cameras in CAMERA_ORDER.
    reordered, cams = {}, iter(CAMERA_ORDER)
    for key, value in features.items():
        if key in CAMERA_ORDER:
            key = next(cams)
            value = features[key]
        reordered[key] = value
    info["features"] = reordered
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    tmp_root.rename(dst_root)
    print(f"Built {dst_root} with cameras {[k for k in reordered if reordered[k]['dtype'] == 'video']}")


if __name__ == "__main__":
    main()
