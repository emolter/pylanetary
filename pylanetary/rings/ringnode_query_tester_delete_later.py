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
    epoch passed in as string
    
'''

import numpy as np

def test_prints(systemtable, bodytable, ringtable):
    
    print('----------------------------------------------')
    print('System-wide data')
    for key in systemtable.keys():
        print(key, systemtable[key])
    print('----------------------------------------------')
    print('Small moons data')
    print(bodytable)
    print('----------------------------------------------')
    print('Individual ring data')
    #print(ringtable.loc['ring','Epsilon']['ascending node'])
    print(ringtable)
    

import astropy.units as u
from astropy.coordinates import EarthLocation, Angle
from astropy.time import Time

from astroquery.solarsystem.pds import RingNode

from nirc2_reduce import image
import matplotlib.pyplot as plt

# fix relative imports later
from core import Ring

infile = '/Users/emolter/research/alma/data/rings_paper/fitsimages/band3_all_briggs0.fits'
alma_coords = (-23.029 * u.deg, -67.755 * u.deg, 5000 * u.m) #lat, lon, alt(m)
alma_coords_astropy = EarthLocation(alma_coords[0], alma_coords[1], alma_coords[2])
alma_image = image.Image(infile)
#epoch = ' '.join(alma_image.header['DATE-OBS'].split('T'))[:16]
epoch = '2008-05-08 22:39'
print('epoch', epoch)
epoch_astropy = Time(epoch, format = 'iso', scale = 'utc')

# send query to Planetary Ring Node
node = RingNode()
bodytable, ringtable = node.ephemeris(
            planet="Uranus", epoch=epoch_astropy, location=alma_coords_astropy, cache=False)
systemtable = bodytable.meta
#test_prints(systemtable, bodytable, ringtable)



# non-changing Epsilon ring params
a = 51149 #km
T = 1 #does not matter
tau = 0.0 # time of periapsis passage, does not matter for now
e = 0.00794

imsize = 300 #pixels
pixscale = 500 #km

'''
Two problems to solve:
1. ring model does not account for eccentricity when projecting to 2-D array
2. when accounting for eccentricity, need to make Uranus at one focus, not at center
'''

# params that do change
omega = systemtable['sub_obs_lon'] + ringtable.loc['ring','Epsilon']['ascending node'].value
#i = 90 - systemtable['opening_angle'].value
i = 90 + systemtable['opening_angle'].value
w = ringtable.loc['ring','Epsilon']['pericenter']

# custom play with these values
#w = 45
#e = 0.4

ringmodel = Ring(a, e, omega, i, w)
img = ringmodel.as_2d_array((imsize, imsize), pixscale) #shape (pixels), pixscale (km)

fig, ax = plt.subplots(1,1, figsize = (9,9))

ax.imshow(img, origin = 'lower')
ax.set_xlabel(r'Distance (10$^3$ km)')
ax.set_ylabel(r'Distance (10$^3$ km)')
ax.scatter([imsize/2],[imsize/2], color = 'cyan', marker = '*', s = 50, label = 'center of Uranus')
plt.savefig('example_epsilon_ring.png')
plt.show()