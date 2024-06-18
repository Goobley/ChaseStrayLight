from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
import sunpy.map
from tqdm import tqdm
from weno4 import weno4

@dataclass
class StrayLightTable:
    radii: u.Quantity[u.arcsec]
    wavelength_idxs: Iterable[int]
    data: u.Quantity[u.DN]


@u.quantity_input
def stray_light_table(
    cube: Any,
    min_rad: u.arcsec = 975 * u.arcsec,
    max_rad: u.arcsec = 1120 * u.arcsec,
    rad_step: u.arcsec = 5 * u.arcsec,
    wavelength_idxs : Optional[Iterable[int]] = None,
    quiet: bool = False,
) -> StrayLightTable:
    """
    Computes the off-limb stray light in a CHASE cube.

    This function computes the median of concentric circles of increasing radius
    around the disk at each wavelength in the provided cube. Zeros (outside the
    raster), are filtered before the median is computed.

    Parameters
    ----------
    cube : sunraster.SpectrogramCube or equivalent NDCube
        The data cube
    min_rad : u.arcsec, optional
        The minimum radius to consider. Default 975"
    max_rad : u.arcsec, optional
        The maximum radius. Default 1120"
    rad_step : u.arcsec, optional
        The steps between concentric circles. Default 5"
    wavelength_idxs : iterable of int, optional
        The wavelength indices at which to compute the stray light. Default None
        (all).
    quiet : bool, optional
        Whether to use tqdm to print progress. Default True

    Returns
    -------

    stray_light : StrayLightTable
        Data class containing wavelength radii, wavelength indices and data of
        shape [radii, wavelength_idx].
    """

    radii = np.arange(
        min_rad.to(u.arcsec).value,
        max_rad.to(u.arcsec).value,
        rad_step.to(u.arcsec).value
    ) << u.arcsec

    if wavelength_idxs is None:
        wavelength_idxs = range(cube.data.shape[0])

    stray_light = np.zeros((radii.shape[0], len(wavelength_idxs)))

    rad_iter = radii
    if not quiet:
        rad_iter = tqdm(radii)
    for rad_idx, rad in enumerate(rad_iter):
        rad <<= u.arcsec
        coord_list = [[rad * np.cos(x), rad * np.sin(x)] for x in np.linspace(0, 2.0*np.pi, 360)]
        coords = SkyCoord(coord_list, frame=frames.Helioprojective)
        pix_path = sunpy.map.pixelate_coord_path(cube[71], coords)
        for la in wavelength_idxs:
            intensity = sunpy.map.sample_at_coords(cube[la], pix_path).value
            stray_light[rad_idx, la] = np.median(intensity[intensity > 0])
    return StrayLightTable(
        radii=radii,
        wavelength_idxs=wavelength_idxs,
        data=stray_light << (cube.unit)
    )

def subtract_stray_light(
    cube: Any,
    stray_light: StrayLightTable,
    min_rad: u.arcsec = 975 * u.arcsec,
    quiet: bool = False,
):
    """Subtracts stray light from the cube using a previously computed table.

    Updates cube in place

    """

    spatial_coords = cube.axis_world_coords()[0]
    radii = np.sqrt(spatial_coords.Tx**2 + spatial_coords.Ty**2)
    rad_mask = radii >= min_rad
    zeros_mask = cube[0].data == 0
    mask = rad_mask & ~zeros_mask
    flat_radii = radii[mask].value.reshape(-1)
    idxs = stray_light.wavelength_idxs
    if not quiet:
        idxs = tqdm(idxs)
    for la in idxs:
        stray_corr = np.interp(flat_radii, stray_light.radii.value, stray_light.data[:, la].value)
        cube.data[la][mask] -= stray_corr.astype(np.int16)

    return cube

