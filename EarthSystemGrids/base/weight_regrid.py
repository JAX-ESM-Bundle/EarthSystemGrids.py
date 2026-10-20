"""
NumPy/SciPy counterpart of `esmf_regrid.ESMFRegridder` for offline
preprocessing: float64 throughout and no JAX import.
"""

import numpy as np
import scipy.sparse
import xarray as xr


class WeightRegridder:
    """Applies an ESMF_RegridWeightGen weight file to flat (n_a,) arrays,
    in float64 (scipy.sparse). Destination cells no source cell covers
    (frac_b == 0) come back NaN."""

    def __init__(self, weight_file):
        with xr.open_dataset(weight_file) as ds:
            self.n_a = ds.sizes["n_a"]
            self.n_b = ds.sizes["n_b"]
            self.matrix = scipy.sparse.csr_matrix(
                (ds["S"].values, (ds["row"].values - 1, ds["col"].values - 1)),
                shape=(self.n_b, self.n_a),
            )
            self.frac_b = ds["frac_b"].values
            self.area_a = ds["area_a"].values
            self.area_b = ds["area_b"].values
        self.weight_file = str(weight_file)

    def __call__(self, field_a):
        field_a = np.asarray(field_a, dtype=np.float64)
        if field_a.shape != (self.n_a,):
            raise ValueError(f"expected a flat ({self.n_a},) field for {self.weight_file}, got {field_a.shape}")
        field_b = self.matrix @ field_a
        field_b[self.frac_b <= 0] = np.nan
        return field_b
