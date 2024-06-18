from load_cube import load_cube
from stray_light import stray_light_table, subtract_stray_light
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import frames
import numpy as np
import matplotlib.pyplot as plt

cube = load_cube("RSM20240303T130012_0021_HA.fits")
coord = SkyCoord(-540 * u.arcsec, -845 * u.arcsec, frame=frames.Helioprojective)
uncleaned = np.copy(cube[:, *cube[71].wcs.world_to_array_index(coord)].data)


stray_light = stray_light_table(cube)
subtract_stray_light(cube, stray_light)

cleaned = np.copy(cube[:, *cube[71].wcs.world_to_array_index(coord)].data)

plt.ion()
fig, ax = plt.subplot_mosaic("AB", per_subplot_kw={"A": {"projection": cube[71].wcs}})
cube[71].plot(axes=ax["A"])
lims = cube[71].wcs.world_to_pixel(SkyCoord([-600, -500]*u.arcsec, [-860, -800]*u.arcsec, frame=frames.Helioprojective))
ax["A"].set_xlim(*lims[0])
ax["A"].set_ylim(*lims[1])
ax["A"].plot_coord(coord, "rx")

ax["B"].plot(cube.spectral_axis.value * 1e9, uncleaned, label="Non-corrected")
ax["B"].plot(cube.spectral_axis.value * 1e9, cleaned, label="Corrected")
ax["B"].legend()
