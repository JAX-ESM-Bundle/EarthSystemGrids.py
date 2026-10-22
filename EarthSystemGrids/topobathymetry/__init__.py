"""
Topography/bathymetry and land-sea mask generation from GEBCO for arbitrary
model grids. The step-by-step driver scripts live in workflows/topobathymetry/.
"""

from EarthSystemGrids.topobathymetry.cleanup import (
    BAB_EL_MANDEB,
    BOSPORUS,
    GIBRALTAR,
    HUDSON_STRAIT,
    INDONESIAN_THROUGHFLOW,
    Strait,
    cap_north,
    carve_strait,
    clean_surface,
    derive_fields,
    fill_isolated_ocean,
)
from EarthSystemGrids.topobathymetry.gebco import (
    build_0p1deg_mesh,
    coarsen_gebco_to_0p1deg,
    node_centered_grid,
    open_gebco,
    to_mesh_order,
)
from EarthSystemGrids.topobathymetry.io import global_mean, read_grid_fields, read_mesh, write_grid_fields
