import numpy as np
from matplotlib.path import Path
from typing import Optional, List, Tuple

def polygon_mask(freq_bins: int,
                 time_frames: int,
                 total_time: float,
                 max_fre: float,
                 polygon_points: List[Tuple[float, float]]) -> np.ndarray:
    time_coords = np.linspace(0, total_time, time_frames)
    freq_coords = np.linspace(0, max_fre, freq_bins)
    time_grid, freq_grid = np.meshgrid(time_coords, freq_coords)

    points = np.column_stack((time_grid.ravel(), freq_grid.ravel()))
    poly_path = Path(polygon_points)

    mask = poly_path.contains_points(points)
    return mask.reshape(freq_bins, time_frames)

def reconstruct_from_polygon(tfr: np.ndarray,
                             fs: float,
                             signal_len: int,
                             polygon_points: List[Tuple[float, float]]
                             ) -> Optional[np.ndarray]:
    if tfr is None or polygon_points is None or len(polygon_points) < 3:
        return None

    freq_bins, time_frames = tfr.shape
    total_time = signal_len / fs
    max_fre = fs / 2.0

    mask = polygon_mask(freq_bins, time_frames, total_time, max_fre, polygon_points)

    partial_sum = np.sum(tfr * mask, axis=0)
    return np.real(partial_sum)