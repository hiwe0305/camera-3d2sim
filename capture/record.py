from dataclasses import dataclass
from typing import Iterable
import numpy as np
import imageio.v3 as iio


@dataclass
class Frame:
    index: int
    timestamp: float
    color: np.ndarray
    depth: np.ndarray


def record_session(frames: Iterable[Frame], intrinsics: dict, session, manifest: dict,
                    max_frames: int | None = None) -> dict:
    count = 0
    first_ts = None
    last_ts = None
    for frame in frames:
        if max_frames is not None and count >= max_frames:
            break
        if first_ts is None:
            first_ts = frame.timestamp
        last_ts = frame.timestamp
        iio.imwrite(session.raw_rgb / f"{frame.index:06d}.png", frame.color)
        iio.imwrite(session.raw_depth / f"{frame.index:06d}.png", frame.depth)
        count += 1

    manifest["hardware"]["intrinsics"] = intrinsics
    manifest["capture_stats"]["n_frames"] = count
    manifest["capture_stats"]["duration_s"] = round((last_ts - first_ts), 6) if count > 1 else 0.0
    return manifest
