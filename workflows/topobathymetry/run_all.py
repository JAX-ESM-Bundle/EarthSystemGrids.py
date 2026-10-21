"""
Run all of Part 2 from a YAML config (default: ESG_config.yaml next to this
script): generate the atm and surface grids as SCRIP files from this repo's
grid code, then hand them to run_part2.sh.

Needs Part 1's GEBCO_bathymetry_res-0p1.nc, and ESMF_RegridWeightGen +
mpirun on PATH (e.g. the `jax` conda env). Relative paths in the config are
taken from the current directory.
"""

import argparse
import subprocess
from pathlib import Path

import yaml

from EarthSystemGrids.GaussianLatLon import (
    equally_spaced_bounds,
    gaussian_latitude_bounds,
    generate_mesh,
    offset_equally_spaced_bounds,
)
from EarthSystemGrids.topobathymetry import cleanup

SCRIPT_DIR = Path(__file__).resolve().parent

LAT_SPACINGS = {
    "equal": lambda n: equally_spaced_bounds(n, -90.0, 90.0),
    "offset": lambda n: offset_equally_spaced_bounds(n, -90.0, 90.0),
    "gaussian": gaussian_latitude_bounds,
}

NAMED_STRAITS = {
    s.name: s
    for s in (
        cleanup.INDONESIAN_THROUGHFLOW,
        cleanup.GIBRALTAR,
        cleanup.BAB_EL_MANDEB,
        cleanup.BOSPORUS,
        cleanup.HUDSON_STRAIT,
    )
}


def _lat_lon_bounds(cfg):
    if cfg["lat_spacing"] not in LAT_SPACINGS:
        raise ValueError(f"unknown lat_spacing {cfg['lat_spacing']!r} (expected one of {list(LAT_SPACINGS)})")
    nlon = cfg["nlon"]
    if cfg.get("lon_centered", False):
        lon = equally_spaced_bounds(nlon, -180.0 / nlon, 360.0 - 180.0 / nlon)
    else:
        lon = equally_spaced_bounds(nlon, 0.0, 360.0)
    return LAT_SPACINGS[cfg["lat_spacing"]](cfg["nlat"]), lon


def build_grid(cfg):
    """(label, mesh) for one entry of the config's `grids` section."""
    kind = cfg["type"]
    if kind == "JCMGrid":
        from EarthSystemGrids.JCMGrid import generate_JCMGrid

        return f"JCM_T{cfg['resolution']}", generate_JCMGrid(cfg["resolution"])
    if kind == "RotatedGaussianLatLon":
        from EarthSystemGrids.RotatedGaussianLatLon import RotatedGaussianLatLon

        lat, lon = _lat_lon_bounds(cfg)
        return kind, RotatedGaussianLatLon.generate_mesh(
            lat=lat,
            lon=lon,
            rotation_axis_longitude_deg=cfg["rotation_axis_longitude_deg"],
            rotation_deg=cfg["rotation_deg"],
        )
    if kind == "GaussianLatLon":
        return kind, generate_mesh(*_lat_lon_bounds(cfg))
    raise ValueError(f"unknown grid type {kind!r} (expected JCMGrid, RotatedGaussianLatLon or GaussianLatLon)")


def strait_arg(entry):
    """04_ensure_featrues_of_topobathymetry_in_surface_grid.py --strait text for a strait name or a full spec dict."""
    if isinstance(entry, str):
        if entry not in NAMED_STRAITS:
            raise ValueError(f"unknown strait {entry!r} (named straits: {list(NAMED_STRAITS)})")
        s = NAMED_STRAITS[entry]
        box, seed_a, seed_b = (s.lon_min, s.lon_max, s.lat_min, s.lat_max), s.seed_a, s.seed_b
        name = s.name
    else:
        name, box, seed_a, seed_b = entry["name"], entry["box"], entry["seed_a"], entry["seed_b"]
    join = lambda values: ",".join(str(v) for v in values)
    return f"{name}:{join(box)}:{join(seed_a)}:{join(seed_b)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=SCRIPT_DIR / "ESG_config.yaml", help="YAML config (default: %(default)s)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    scrip = {}
    for role in ("atm", "sfc"):
        label, mesh = build_grid(config["grids"][role])
        scrip[role] = output_dir / f"{label}.SCRIP.nc"
        print(f"Generating {role} grid: {label} -> {scrip[role]}")
        mesh.write_to_SCRIP_grid_file(str(scrip[role]), flatten=True)

    clean = config["cleanup"]
    cleanup_args = ["--land-threshold", str(clean["land_threshold"]), "--min-depth", str(clean["min_depth"])]
    straits = clean.get("straits") or []
    if straits:
        for entry in straits:
            cleanup_args += ["--strait", strait_arg(entry)]
    else:
        cleanup_args.append("--no-straits")

    cmd = [
        str(SCRIPT_DIR / "run_part2.sh"),
        "--atm-scrip", str(scrip["atm"]),
        "--sfc-scrip", str(scrip["sfc"]),
        "--output-dir", str(output_dir),
        "--nproc", str(config["nproc"]),
        "--input-0p1", str(config["input_0p1"]),
        "--", *cleanup_args,
    ]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)
