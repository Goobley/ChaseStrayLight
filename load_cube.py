import astropy.units as u
import os
from astropy.io import fits
from astropy.wcs import WCS
from typing import Optional, Union
from sunraster import SpectrogramCube

def load_cube(path: Union[str, bytes, os.PathLike], unit: Optional[u.Unit] = u.DN):
    """Load a CHASE cube from fits.

    Parameters
    ----------
    path : str, bytes, os.PathLike
        The cube path
    unit : astropy unit
        The unit to attach to the data. Default: DN. None is valid.

    Returns
    -------
    cube : sunraster.SpectrogramCube
        A sunraster cube for this observation
    """

    data = fits.open(path)
    data[1].header["CTYPE1"] = "HPLT-TAN"
    data[1].header["CTYPE2"] = "HPLN-TAN"
    data[1].header["CTYPE3"] = "WAVE"
    del data[1].header["RSUN_REF"]
    wcs = WCS(data[1].header)

    cube = SpectrogramCube(data[1].data, wcs, unit=unit)
    return cube
