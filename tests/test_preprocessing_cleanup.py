"""
Tests for preprocessing_workflow's Part 2 surface cleanup, on a small
synthetic global lat-lon grid (10 x 36, 18deg x 10deg cells).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing_workflow"))

from common import face_adjacency, read_scrip  # noqa: E402
from surface_cleanup import (  # noqa: E402
    Strait,
    carve_strait,
    clean_surface,
    derive_fields,
    fill_isolated_ocean,
)

from EarthSystemGrids.GaussianLatLon import equally_spaced_bounds, generate_mesh  # noqa: E402

NLAT, NLON = 10, 36


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    mesh = generate_mesh(equally_spaced_bounds(NLAT, -90.0, 90.0), equally_spaced_bounds(NLON, 0.0, 360.0))
    path = tmp_path_factory.mktemp("grid") / "test.SCRIP.nc"
    mesh.write_to_SCRIP_grid_file(path, flatten=True)
    return read_scrip(path)


@pytest.fixture(scope="module")
def adjacency(grid):
    return face_adjacency(grid)


def idx(j, i):
    return j * NLON + i


def test_adjacency_is_four_neighbour_with_periodic_seam(grid, adjacency):
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    # pole rows lose their polar (degenerate) neighbour; no cell connects across the pole
    assert np.all(degree.reshape(NLAT, NLON)[1:-1] == 4)
    assert np.all(degree.reshape(NLAT, NLON)[[0, -1]] == 3)
    assert adjacency[idx(5, 0), idx(5, NLON - 1)] == 1
    assert adjacency[idx(5, 0), idx(4, 0)] == 1
    assert adjacency[idx(5, 0), idx(4, 1)] == 0


def test_fill_isolated_ocean(adjacency):
    mask = np.zeros(NLAT * NLON, dtype=np.int32)
    mask[[idx(j, 0) for j in range(NLAT)]] = 1          # a meridional wall ...
    mask[[idx(j, 1) for j in range(NLAT)]] = 1          # ... two cells wide
    for j in (3, 4, 5):                                 # a one-cell sea at (4, 10), ringed by land
        for i in (9, 10, 11):
            mask[idx(j, i)] = 1
    mask[idx(4, 10)] = 0

    area = np.ones(mask.size)
    new, filled = fill_isolated_ocean(adjacency, mask, area)
    assert list(filled) == [idx(4, 10)]
    assert new[idx(4, 10)] == 1
    assert np.array_equal(np.delete(new, filled), np.delete(mask, filled))


def test_carve_strait_opens_blocked_passage(grid, adjacency):
    # Land wall at i=17..18 on every row, with ocean basins either side
    mask = np.zeros(NLAT * NLON, dtype=np.int32)
    for j in range(NLAT):
        for i in (17, 18):
            mask[idx(j, i)] = 1
    land_fraction = mask.astype(float)
    land_fraction[idx(5, 17)] = 0.6                    # the marginal cell the path should prefer
    strait = Strait("test", 145.0, 195.0, -30.0, 30.0, (150.0, 5.0), (190.0, 5.0))

    new, carved = carve_strait(grid, adjacency, mask, land_fraction, strait)
    assert set(carved) == {idx(5, 17), idx(5, 18)}
    assert np.all(new[carved] == 0)

    again, carved_again = carve_strait(grid, adjacency, new, land_fraction, strait)
    assert carved_again.size == 0
    assert np.array_equal(again, new)


def test_derive_fields_consistency():
    elevation = np.array([100.0, -50.0, 5.0, -3.0])
    mask = np.array([1, 1, 0, 0])
    topo, bathy = derive_fields(elevation, mask, min_depth=10.0)
    assert np.array_equal(topo, [100.0, 0.0, 0.0, 0.0])
    assert np.array_equal(bathy, [0.0, 0.0, -10.0, -10.0])


def test_clean_surface_flips_land_fraction_only_where_edited(grid):
    land_fraction = np.zeros(NLAT * NLON)
    land_fraction[[idx(j, 0) for j in range(NLAT)]] = 0.9
    for j in (3, 4, 5):
        for i in (9, 10, 11):
            land_fraction[idx(j, i)] = 0.8
    land_fraction[idx(4, 10)] = 0.2
    elevation = np.where(land_fraction >= 0.5, 200.0, -1000.0)

    fields, report = clean_surface(
        grid, np.ones(land_fraction.size), elevation, land_fraction,
        land_threshold=0.5, min_depth=10.0, straits=[],
    )
    assert list(report["filled:isolated_ocean"]) == [idx(4, 10)]
    assert fields["land_fraction"][idx(4, 10)] == 1.0
    assert fields["landsea_mask"][idx(4, 10)] == 1
    unchanged = np.ones(land_fraction.size, bool)
    unchanged[idx(4, 10)] = False
    assert np.array_equal(fields["land_fraction"][unchanged], land_fraction[unchanged])
    # filled cell had negative elevation: land now, so topography 0 and bathymetry 0
    assert fields["topography"][idx(4, 10)] == 0.0
    assert fields["bathymetry"][idx(4, 10)] == 0.0


def test_strait_box_can_cross_zero_longitude(grid):
    from surface_cleanup import _in_box

    strait = Strait.parse("wrap:350,10,-10,10:355,0:5,0")
    lon = np.mod(grid["center_lon"][_in_box(grid, strait)], 360.0)
    assert set(np.round(lon)) == {355.0, 5.0}
