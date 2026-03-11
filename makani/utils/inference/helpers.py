# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from more_itertools import batched, divide
from typing import Optional, List, Iterator, Tuple
import datetime as dt

import numpy as np
import torch.utils.data as tud


def split_list(lst: List[int], nchunks: int) -> List[List[int]]:
    return [list(x) for x in list(divide(nchunks, lst))]


class SortedIndexSampler(tud.Sampler[List[int]]):
    def __init__(self, indices: List[int], maxind: int, batch_size: int, rollout_steps: int, rollout_dt: int, incomplete_rollouts: Optional[bool] = False) -> None:

        # make sure the batch size is sane
        batch_size = min(len(indices), batch_size)
        batches = map(list, batched(indices, batch_size))
        self.indices = []
        for batch in batches:
            rollout = []
            append = True
            for s in range(0, rollout_steps+1):
                shift = [b + rollout_dt * s for b in batch]
                if max(shift) >= maxind:
                    append = False
                    break

                rollout.append(shift)

            if append or incomplete_rollouts:
                self.indices += rollout

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.indices:
            yield batch


class SimpleIndexSampler(tud.Sampler[List[int]]):
    def __init__(self, indices: List[List[int]]) -> None:
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.indices:
            yield batch


def translate_date_sampler_to_timedelta_sampler(sampler, date_dataset, timedelta_dataset):
    indexlist = []
    iterator = iter(sampler)
    for indices in iterator:
        tstamps = [date_dataset.get_time_at_index(idx) for idx in indices]
        timedeltas = [t - dt.datetime(year=t.year, month=1, day=1, hour=0, minute=0, second=0, tzinfo=dt.timezone.utc) for t in tstamps]
        indexlist.append([timedelta_dataset.get_index_at_time(t) for t in timedeltas])

    return SimpleIndexSampler(indexlist)


def compute_crop_indices(
    lat_lon: Tuple[List[float], List[float]],
    crop_region: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Resolve a geographic bounding box to grid index arrays.

    Both lat and lon indices are returned as arrays (not slices) because
    longitude wrapping can produce non-contiguous indices.

    Parameters
    ============
    lat_lon : tuple of (lats, lons)
        1-D arrays of latitude and longitude values. Latitudes may be
        N-to-S or S-to-N. Longitudes may use 0-360 or -180/180 convention.
    crop_region : tuple of (min_lat, max_lat, min_lon, max_lon)
        Inclusive geographic bounding box. Latitude bounds are auto-sorted.
        If min_lon > max_lon the selection wraps around
        (e.g. (270, 90) selects 270->360->0->90).

    Returns
    ============
    lat_indices : np.ndarray
        Integer indices into lats that fall within the latitude bounds,
        preserving the original N-to-S or S-to-N grid ordering.
    lon_indices : np.ndarray
        Integer indices into lons that fall within the longitude bounds,
        in ascending index order (preserving original grid ordering).
        Crop region must use the same coordinate convention as the grid
        (e.g. 0-360 or -180-180).
    """

    lats, lons = map(np.array, lat_lon)
    lat_crop, lon_crop = crop_region[:2], crop_region[2:]

    # --- latitude indices ---
    min_lat, max_lat = min(lat_crop), max(lat_crop)
    lat_mask = (lats >= min_lat) & (lats <= max_lat)
    lat_indices = np.nonzero(lat_mask)[0]

    # --- longitude indices ---
    min_lon, max_lon = lon_crop
    if min_lon <= max_lon:
        # Simple contiguous range
        lon_mask = (lons >= min_lon) & (lons <= max_lon)
        lon_indices = np.nonzero(lon_mask)[0]
    else:
        # Wrapping range: select head (<= max_lon) | tail (>= min_lon)
        lon_mask = (lons >= min_lon) | (lons <= max_lon)
        lon_indices = np.nonzero(lon_mask)[0]

    return lat_indices, lon_indices


def compute_local_crop(
    global_indices: Tuple[np.ndarray, np.ndarray],
    local_offset: Tuple[int, int],
    local_size: Tuple[int, int],
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[slice, slice]]:
    r"""
    Given global crop indices, compute where to read from a rank's local
    buffer and where to write in the cropped output file.

    Each rank owns a contiguous spatial tile of the global grid. This function
    intersects the crop region with that tile to produce buffer indices for
    reading and output slices for writing.

    When a rank's tile has no overlap with the crop region, both output slices
    are empty (start == stop) and the buffer indices are empty arrays.

    Parameters
    ============
    global_indices : tuple of (lat_indices, lon_indices)
        Arrays from compute_crop_indices, representing the global grid
        indices that need to be output.
    local_offset : tuple of (lat_offset, lon_offset)
        Start of this rank's tile in the global grid.
    local_size : tuple of (lat_size, lon_size)
        Size of this rank's tile.

    Returns
    ============
    local_indices : tuple of np.ndarray
        np.ix_-shaped index arrays for rectangular selection from the rank's
        local buffer. Shapes are (n_lat, 1) and (1, n_lon). These are arrays
        rather than slices because longitude wrapping can produce non-contiguous
        buffer indices (e.g. [350, ..., 359, 0, ..., 30]).
    output_indices : tuple of slice
        Contiguous slices into the cropped output dimensions. Always contiguous
        because a rank's tile occupies a contiguous block within the global
        crop index array.
    """
    local_indices = []
    output_indices = []
    for global_idx, offset, size in zip(global_indices, local_offset, local_size):
        mask = (global_idx >= offset) & (global_idx < offset + size)
        out_start, out_count = int(np.argmax(mask)), int(np.sum(mask))
        output_indices.append(slice(out_start, out_start + out_count))
        local_indices.append(global_idx[mask] - offset)
        
        # sanity check: output slice covers only selected indices
        assert mask[output_indices[-1]].all()

    local_indices = np.ix_(*local_indices)
    assert len(local_indices) == 2
    return local_indices, tuple(output_indices)
