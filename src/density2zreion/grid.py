import os

import h5py
import numpy as np
from tqdm import tqdm


class BaseDensityGrid:
    def __init__(self, snapshot_file, grid_file, N_max_load=None, N_cell=None):
        self._read_header(snapshot_file)

        self._N_max_load = N_max_load or self._N_part

        self._N_cell = N_cell or np.round(self._box_size / (np.pi * 1e3)).astype(int)
        self._cell_size = self._box_size / self._N_cell

        self._grid, self._N_part_total = self._get_grid(grid_file)
        self._N_part_total += self._N_part

        for i in tqdm(
            range(np.ceil(self._N_part / self._N_max_load).astype(int)),
            desc="Assigning particles",
        ):
            _idx_start = i * self._N_max_load
            _idx_end = min((i + 1) * self._N_max_load, self._N_part)

            part_coords, part_masses = self._load_particles(
                snapshot_file, slice(_idx_start, _idx_end)
            )
            self._assign_particles(part_coords, part_masses)

    def _read_header(self, snapshot_file):
        with h5py.File(snapshot_file, "r") as f:
            _header = f["Header"]

            _h = _header.attrs.get("HubbleParam", 1.0)

            self._box_size = _header.attrs["BoxSize"] / _h
            self._N_part = _header.attrs["NumPart_ThisFile"][1]

            self._z = _header.attrs["Redshift"]
            self._time = _header.attrs["Time"]

    def _load_particles(self, snapshot_file, idx_range):
        with h5py.File(snapshot_file, "r") as f:
            _h = f["Header"].attrs.get("HubbleParam", 1.0)

            part_coords = f["PartType1"]["Coordinates"][idx_range] / _h
            part_masses = f["Header"].attrs["MassTable"][1] / _h
            part_masses = part_masses or f["PartType1"]["Masses"][idx_range] * 1e10 / _h

        return part_coords, part_masses

    def _get_grid(self, grid_file):
        if os.path.exists(grid_file):  # noqa: PTH110
            with h5py.File(grid_file, "r") as f:
                grid = f["DensityDM"][:]
                N_part_total = f["Header"].attrs["NumPart"]
        else:
            grid = np.zeros(
                (self._N_cell, self._N_cell, self._N_cell), dtype=np.float64
            )
            N_part_total = np.int32(0)

        return grid, N_part_total

    def _assign_particles(self, part_coords, part_masses):
        _msg = "Not implemented in base class."
        raise NotImplementedError(_msg)

    def save_grid(self, grid_file):
        with h5py.File(grid_file, "w") as f:
            _header = f.create_group("Header")

            _header.attrs.create("BoxSize", self._box_size, dtype=np.float64)
            _header.attrs.create("NumPart", self._N_part_total, dtype=np.int32)
            _header.attrs.create("NumCells", self._N_cell, dtype=np.int32)
            _header.attrs.create("CellSize", self._cell_size, dtype=np.float64)
            _header.attrs.create("Redshift", self._z, dtype=np.float64)
            _header.attrs.create("Time", self._time, dtype=np.float64)

            f.create_dataset("DensityDM", data=self._grid, dtype=np.float64)


class NGPDensityGrid(BaseDensityGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _assign_particles(self, part_coords, part_masses):
        _part_densities = part_masses / self._cell_size**3

        _norm_part_coords = part_coords / self._cell_size - 0.5
        _part_cell_indices = np.round(_norm_part_coords).astype(int)

        np.add.at(
            self._grid,
            (
                _part_cell_indices[:, 0] % self._N_cell,
                _part_cell_indices[:, 1] % self._N_cell,
                _part_cell_indices[:, 2] % self._N_cell,
            ),
            _part_densities,
        )


class CICDensityGrid(BaseDensityGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _assign_particles(self, part_coords, part_masses):
        _part_densities = part_masses / self._cell_size**3

        _norm_part_coords = part_coords / self._cell_size - 0.5
        _part_cell_indices = np.floor(_norm_part_coords).astype(int)
        _part_cell_offsets = _norm_part_coords - _part_cell_indices

        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    _weights = (
                        (1 - dx + (2 * dx - 1) * _part_cell_offsets[:, 0])
                        * (1 - dy + (2 * dy - 1) * _part_cell_offsets[:, 1])
                        * (1 - dz + (2 * dz - 1) * _part_cell_offsets[:, 2])
                    )

                    np.add.at(
                        self._grid,
                        (
                            (_part_cell_indices[:, 0] + dx) % self._N_cell,
                            (_part_cell_indices[:, 1] + dy) % self._N_cell,
                            (_part_cell_indices[:, 2] + dz) % self._N_cell,
                        ),
                        _weights * _part_densities,
                    )
