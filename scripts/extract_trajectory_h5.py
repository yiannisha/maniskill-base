#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a ManiSkill trajectory.h5 into a readable folder tree."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--trajectory",
        type=Path,
        help="Path to trajectory.h5",
    )
    group.add_argument(
        "--experiment-dir",
        type=Path,
        help="Experiment directory containing trajectory.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the extracted files. Defaults to <experiment>/trajectory_extracted",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the output directory if it already exists.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    trajectory_path = args.trajectory
    if trajectory_path is None:
        trajectory_path = args.experiment_dir / "trajectory.h5"
    trajectory_path = trajectory_path.expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory.h5 not found: {trajectory_path}")

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_dir = trajectory_path.parent / "trajectory_extracted"
    return trajectory_path, output_dir


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")


def depth_to_preview(frame: np.ndarray) -> np.ndarray:
    depth = np.asarray(frame)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = depth.astype(np.float32)

    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return preview

    lo = float(depth[valid].min())
    hi = float(depth[valid].max())
    if hi <= lo:
        preview[valid] = 255
        return preview

    normalized = (depth - lo) / (hi - lo)
    preview[valid] = np.clip(normalized[valid] * 255.0, 0, 255).astype(np.uint8)
    return preview


def save_rgb_sequence(base_path: Path, array: np.ndarray) -> dict[str, Any]:
    np.save(base_path.with_suffix(".npy"), array)
    frames_dir = base_path.parent / f"{base_path.name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(array):
        iio.imwrite(frames_dir / f"frame_{idx:06d}.png", frame)
    return {
        "raw_npy": str(base_path.with_suffix(".npy").name),
        "frames_dir": frames_dir.name,
        "num_frames": int(array.shape[0]),
    }


def save_depth_sequence(base_path: Path, array: np.ndarray) -> dict[str, Any]:
    np.save(base_path.with_suffix(".npy"), array)
    preview_dir = base_path.parent / f"{base_path.name}_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(array):
        preview = depth_to_preview(frame)
        iio.imwrite(preview_dir / f"frame_{idx:06d}.png", preview)
    return {
        "raw_npy": str(base_path.with_suffix(".npy").name),
        "preview_dir": preview_dir.name,
        "num_frames": int(array.shape[0]),
    }


def save_generic_dataset(base_path: Path, array: np.ndarray) -> dict[str, Any]:
    if array.ndim == 0:
        save_json(base_path.with_suffix(".json"), {"value": array.item()})
        return {"raw_json": base_path.with_suffix(".json").name}

    np.save(base_path.with_suffix(".npy"), array)
    return {"raw_npy": base_path.with_suffix(".npy").name}


def extract_dataset(dataset: h5py.Dataset, output_root: Path) -> dict[str, Any]:
    array = dataset[()]
    base_path = output_root / dataset.name.lstrip("/")

    info = {
        "h5_path": dataset.name,
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
    }

    if (
        dataset.name.endswith("/rgb")
        and isinstance(array, np.ndarray)
        and array.ndim == 4
        and array.shape[-1] in (3, 4)
    ):
        info.update(save_rgb_sequence(base_path, array))
    elif (
        dataset.name.endswith("/depth")
        and isinstance(array, np.ndarray)
        and array.ndim in (3, 4)
    ):
        info.update(save_depth_sequence(base_path, array))
    else:
        info.update(save_generic_dataset(base_path, array))

    save_json(base_path.with_suffix(".meta.json"), info)
    return info


def extract_group(group: h5py.Group, output_root: Path, manifest: list[dict[str, Any]]) -> None:
    for _, item in group.items():
        if isinstance(item, h5py.Group):
            (output_root / item.name.lstrip("/")).mkdir(parents=True, exist_ok=True)
            extract_group(item, output_root, manifest)
        elif isinstance(item, h5py.Dataset):
            manifest.append(extract_dataset(item, output_root))


def copy_sidecar_json(trajectory_path: Path, output_dir: Path) -> None:
    json_path = trajectory_path.with_suffix(".json")
    if not json_path.is_file():
        return
    data = json.loads(json_path.read_text(encoding="utf-8"))
    save_json(output_dir / "trajectory_metadata.json", data)


def main() -> int:
    args = parse_args()
    trajectory_path, output_dir = resolve_paths(args)
    ensure_clean_dir(output_dir, overwrite=args.overwrite)

    manifest: list[dict[str, Any]] = []
    copy_sidecar_json(trajectory_path, output_dir)

    with h5py.File(trajectory_path, "r") as h5_file:
        for name, item in h5_file.items():
            if isinstance(item, h5py.Group):
                (output_dir / name).mkdir(parents=True, exist_ok=True)
                extract_group(item, output_dir, manifest)
            elif isinstance(item, h5py.Dataset):
                manifest.append(extract_dataset(item, output_dir))

    save_json(
        output_dir / "manifest.json",
        {
            "source_h5": str(trajectory_path),
            "output_dir": str(output_dir),
            "datasets": manifest,
        },
    )
    print(f"Extracted {len(manifest)} datasets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
