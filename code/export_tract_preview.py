import json
from pathlib import Path

import numpy as np


SOURCE = Path(
    "/Users/leonmartin_bih/tools/bsplot/builds/nQUoDtIPe/0/martinl/bsplot/docs/data/"
    "dTOR_10K_sample_subsampled_10k.tck"
)
TARGET = Path(__file__).resolve().parents[1] / "data" / "tracts_preview.json"


def read_tck(path):
    with path.open("rb") as handle:
        offset = None
        while True:
            line = handle.readline()
            if line.startswith(b"file:"):
                offset = int(line.decode().strip().split()[-1])
            if line == b"END\n":
                break
        handle.seek(offset)
        return np.fromfile(handle, dtype="<f4").reshape(-1, 3)


def streamlines_from_points(points):
    separator_rows = np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1)
    separators = np.where(separator_rows)[0]
    starts = np.r_[0, separators[:-1] + 1]
    return [points[start:stop] for start, stop in zip(starts, separators) if stop - start >= 12]


def resample_streamline(streamline):
    point_count = min(28, max(12, len(streamline) // 4))
    point_indices = np.linspace(0, len(streamline) - 1, point_count, dtype=int)
    return np.round(streamline[point_indices], 3).tolist()


def main():
    points = read_tck(SOURCE)
    streamlines = streamlines_from_points(points)
    selected_indices = np.linspace(0, len(streamlines) - 1, min(480, len(streamlines)), dtype=int)
    valid_points = points[np.isfinite(points).all(axis=1)]
    payload = {
        "source": SOURCE.name,
        "streamline_count": len(streamlines),
        "preview_count": len(selected_indices),
        "bounds": {
            "min": np.round(valid_points.min(axis=0), 3).tolist(),
            "max": np.round(valid_points.max(axis=0), 3).tolist(),
        },
        "streamlines": [resample_streamline(streamlines[index]) for index in selected_indices],
    }
    TARGET.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {TARGET} ({TARGET.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()