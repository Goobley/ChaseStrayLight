from load_cube import load_cube
from stray_light import stray_light_table, subtract_stray_light
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import frames
import numpy as np
import matplotlib.pyplot as plt
import lightweaver as lw
import promweaver as pw
import xarray as xr
import os


def prep_cube(path, roi=None):
    cube = load_cube(path)
    stray_light = stray_light_table(cube)
    subtract_stray_light(cube, stray_light)

    centre = cube.crop(
        [
            SkyCoord(-20 * u.arcsec, -20 * u.arcsec, frame=frames.Helioprojective),
            None,
        ],
        [
            SkyCoord(20 * u.arcsec, 20 * u.arcsec, frame=frames.Helioprojective),
            None
        ]
    )
    average_mu0 = np.mean(centre.data, axis=(1, 2))
    vac_wl = lw.air_to_vac(np.array(cube.spectral_axis) * 1e9)
    # NOTE(cmo): "Calibrate" against FALC
    ctx = pw.compute_falc_bc_ctx(active_atoms=["H"])
    falc_ha = ctx.compute_rays(wavelengths=vac_wl, mus=1.0)
    falc_ha *= 1e9
    # NOTE(cmo): The red continuum side is cleaner in chase than the blue
    calibration_factor = falc_ha[-1] / average_mu0.astype(np.float32)[-1]

    if roi is not None:
        cube = cube.crop([roi[0], None], [roi[1], None])

    axw = cube.axis_world_coords()
    lon = np.ascontiguousarray(axw[0].Tx[0, :].value)
    lat = np.ascontiguousarray(axw[0].Ty[:, 0].value)

    ds = xr.Dataset(
        data_vars={
            "I": (["wavelength", "hp-lat", "hp-lon"], cube.data.astype(np.float32) * np.float32(calibration_factor))
        },
        coords={
            "wavelength": ("wavelength", vac_wl),
            "hp-lon": ("hp-lon", lon),
            "hp-lat": ("hp-lat", lat),
        },
    )

    return ds

if __name__ == "__main__":

    roi = [
        SkyCoord(-960 * u.arcsec, -800 * u.arcsec, frame=frames.Helioprojective),
        SkyCoord(-620 * u.arcsec, -350 * u.arcsec, frame=frames.Helioprojective),
    ]
    # ds = prep_cube("RSM20240303T130012_0021_HA.fits", roi=roi)
    base_path = "/mnt/c/Users/cmo/OneDrive - University of Glasgow/ChaseScoop20240303/"
    files = [base_path + f for f in os.listdir(base_path) if f.endswith(".fits")]
    for file in files:
        ds = prep_cube(file, roi)
        ds.to_netcdf(file[:-5] + "_crop.nc")

