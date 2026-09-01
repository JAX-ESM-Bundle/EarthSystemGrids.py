from EarthSystemGrids.base.sphere_rotation import rotate_lonlat
import xarray as xr
import numpy as np
import argparse

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input ERA5 file", default="EarthSystemGrids/data/ERA5_landsea_mask_original.nc")
    parser.add_argument("--capped-latitude", type=float, help="The size of capped poles", required=True)
    parser.add_argument("--rotation-deg", type=float, help="The rotation of spherical coordinate in degress. Default points to greenland", default=-12)
    parser.add_argument("--axis-longitude-deg", type=float, help="The longitude of the rotation axis along the equator in degress. Default points to greenland", default=48.0)
    parser.add_argument("--aquaplanet", action="store_true", help="If set, then the original lsm is set to 0.")
    parser.add_argument("--flat-topo", action="store_true", help="Set topography z=0. If not set, then use the original one.")
    parser.add_argument("--output", type=str, help="Output modified file", default=None)
    args = parser.parse_args()

  
    capped_latitude = args.capped_latitude
    axis_longitude_deg = args.axis_longitude_deg
    rotation_deg = args.rotation_deg
    output = f"capped_RGLL_lat-{capped_latitude:.1f}.nc" if args.output is None else args.output

    ds = xr.open_dataset(args.input)
    
    lsm = ds["lsm"].isel(valid_time=0).transpose("longitude", "latitude").to_numpy()
    llon, llat = np.meshgrid(np.deg2rad(ds.coords["longitude"]), np.deg2rad(ds.coords["latitude"]))
    
    llon_rotated, llat_rotated = rotate_lonlat(llon, llat, axis_longitude_deg=axis_longitude_deg, rotation_deg=rotation_deg)
    
    lsm_capped = (np.abs(np.rad2deg(llat_rotated)) > capped_latitude).astype(np.float32)
    
    lsm_updated = ds["lsm"].to_numpy()[0]

    if args.aquaplanet:
        print("--aquaplanet is set. Original lsm is disregarded. All set to 0.")
        lsm_updated[:] = 0.0
    else:    
        print("--aquaplanet is not set. Add capping on top of the original lsm.")

    lsm_updated[lsm_capped == 1] = 1.0
    
    ds["lsm"].values[0, :, :] = lsm_updated

    if args.flat_topo:
        print("Flatten topography because --flat-topo is set.")
        ds["z"].values[:] = 0.0
    else:
        print("--flat-topo is not set. Keep the original topography z values.")
    
    print("Write to: ", output)
    ds.to_netcdf(output)
    
    
     
     
