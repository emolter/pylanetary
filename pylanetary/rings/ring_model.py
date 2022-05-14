#!/usr/bin/env python

'''
using this to test fully-built pds.RingNode tool
run the steps below to test

cd ~/Python/astroquery
python setup.py test
python setup.py install
cd ~/Python/pylanetary/pylanetary/rings
./ring_model.py

tested:
    coordinates passed in as tuple
    coordinates passed in as EarthLocation object
    obs_time passed in as string
    
'''

import astropy.units as u
from astropy.coordinates import EarthLocation, Angle
from astropy.time import Time

from astroquery.solarsystem.pds import RingNode

from nirc2_reduce import image
from almatools import rings
import matplotlib.pyplot as plt

infile = '/Users/emolter/research/alma/data/rings_paper/fitsimages/band3_all_briggs0.fits'
alma_coords = (-23.029 * u.deg, -67.755 * u.deg, 5000 * u.m) #lat, lon, alt(m)
alma_coords_astropy = EarthLocation(alma_coords[0], alma_coords[1], alma_coords[2])
alma_image = image.Image(infile)
#obs_time = ' '.join(alma_image.header['DATE-OBS'].split('T'))[:16]
obs_time = '2024-05-08 22:39'
print('obs_time', obs_time)
obs_time_astropy = Time(obs_time, format = 'iso', scale = 'utc')

# send query to Planetary Ring Node
node = RingNode()
systemtable, bodytable, ringtable = node.ephemeris(
            planet="Neptune", obs_time=obs_time_astropy, location=alma_coords_astropy, cache=False)
print(node.uri)
print(systemtable.keys())
print(bodytable.columns)
print(ringtable.columns)

print('----------------------------------------------')
print('System-wide data')
for i, key in enumerate(systemtable):
    print(key, systemtable[key])
print('----------------------------------------------')
print('Small moons data')
print(bodytable)
print('----------------------------------------------')
print('Individual ring data')
#print(ringtable.loc['ring','Epsilon']['ascending node'])
print(ringtable)


# non-changing Epsilon ring params
a = 51.149 #10^3 km
T = 1 #does not matter
tau = 0.0 # time of periapsis passage, does not matter
e = 0.00794

# params that do change
omega = systemtable['sub_obs_lon'] + ringtable.loc['ring','Epsilon']['ascending node'].value
i = 90 - systemtable['opening_angle'].value
w = ringtable.loc['ring','Epsilon']['pericenter'].value


ringmodel = rings.ringModel(a, T, tau, e, omega, i, w, steps = 500)
ringmodel.to_2d_array(150, 150)

fig, ax = plt.subplots(1,1, figsize = (9,9))

ax.imshow(ringmodel.img, origin = 'lower')
ax.set_xlabel(r'Distance 10$^3$ km')
ax.set_ylabel(r'Distance 10$^3$ km')
ax.scatter([75],[75], color = 'cyan', marker = '*', s = 50)
plt.savefig('example_epsilon_ring.png')
