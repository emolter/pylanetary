
from astropy import table
from astroquery.solarsystem.pds import RingNode
from astroquery.solarsystem.jpl import Horizons
from astropy.coordinates import Angle
import astropy.units as u
from astropy import convolution
from astropy.coordinates import EarthLocation, Angle
from astropy.time import Time
from photutils import aperture

import numpy as np
from collections import OrderedDict
from scipy.spatial.transform import Rotation
from scipy import ndimage
import importlib

# fix imports later
from ..utils import *

'''
To implement
------------
* make as_azimuthal_wedges() allow non-square images!
* class that inherits from Ring for asymmetric rings like Uranus's epsilon ring
* make azimuthal wedges code much faster by implementing wedges as a photutils object
* make RingSystemModelObservation account for the peculiar rotation angles of each ring
    relative to the system
'''

horizons_lookup = {
    'Mars': '499',
    'Jupiter': '599',
    'Saturn': '699',
    'Uranus': '799',
    'Neptune': '899',
    'Pluto': '999'}


def vector_normalize(v):
    '''turn vector into unit vector'''
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def vector_magnitude(a):
    '''expects np array of shape (x, 3) or (x, 2)
    corresponding to a bunch of vectors
        this is a bit faster than using np.linalg.norm'''
    if len(a.shape) == 1:
        return np.sum(np.sqrt(a * a))
    return np.sqrt((a * a).sum(axis=1))


def plane_project(x, z):
    '''projects x onto plane where z is normal vector to that plane'''
    z = vector_normalize(z)
    return np.cross(z, np.cross(x, z))


def make_rot(i, omega, w):
    '''use i, omega, and w as Euler rotation angles and return a rotation object'''
    return Rotation.from_euler('zxz', [w, i, omega], degrees=True)


def rotate_and_project(vec, rot, proj_plane=[0, 0, 1]):
    '''
    use i, omega, and w as Euler rotation angles to project a vector
    first into 3-D then onto a 2-D plane
    '''
    return plane_project(rot.apply(vec), proj_plane)


def b_from_ae(a, e):
    return a * np.sqrt(1 - e**2)


def vector_ellipse(u, v, t, origin=np.array([0, 0, 0])):
    '''
    https://math.stackexchange.com/questions/3994666/parametric-equation-of-an-ellipse-in-the-3d-space#:~:text=In%20the%20parametric%20equation%20x,a%20point%20with%20minimum%20curvature.
    u, v are vectors corresponding to the vectorized a, b
    t are the data points from 0 to 2pi
    '''
    u = u[np.newaxis, :]
    v = v[np.newaxis, :]
    t = t[:, np.newaxis]
    # print(u*np.cos(t))

    return origin + u * np.cos(t) + v * np.sin(t)
    
    
def project_ellipse(a,e,i,omega,w,n=100, origin = np.array([0,0,0]), proj_plane = [0,0,1]):
    '''
    make a projection of an ellipse with the given params using i,omega,w as Euler rotation angles
    a: any distance unit
    e: unitless
    i, omega, w: assume degrees
    n: number of points in ellipse circumference
    origin: units of a, expects array
    '''
    
    # simple pre-calculations
    b = a*np.sqrt(1-e**2)
    c = a*e
    f0 = np.array([origin[0] + c, origin[1], 0]) # foci
    f1 = np.array([origin[0] - c, origin[1], 0])
    a_vec = np.array([a,0,0])
    b_vec = np.array([0,b,0])
    
    # apply projections to a, b, f0, f1
    rot = make_rot(i,omega,w)
    f0p = rotate_and_project(f0, rot, proj_plane=proj_plane)
    f1p = rotate_and_project(f1, rot, proj_plane=proj_plane)
    a_vec_p = rotate_and_project(a_vec, rot, proj_plane=proj_plane)
    b_vec_p = rotate_and_project(b_vec, rot, proj_plane=proj_plane)
    
    # make and project ellipse circumference
    t = np.linspace(0,2*np.pi,n)
    ell = vector_ellipse(a_vec, b_vec, t, origin=origin)
    ell_p = rotate_and_project(ell, rot, proj_plane=proj_plane)
    
    # dict of outputs
    output = {'a':a_vec_p,
             'b':b_vec_p,
             'f0':f0p,
             'f1':f1p,
             'ell':ell_p}
    
    return output


def calc_abtheta(ell):
    '''
    given vectors defining the circumference of an ellipse, 
    find corresponding values of a, b, and theta
    using the fact that locations of a, b are max, min of ellipse vectors
    '''
    mag = vector_magnitude(ell)
    a, b = np.max(mag), np.min(mag)
    wherea = np.argmax(mag)
    whereb = np.argmin(mag)
    xa, ya, _ = ell[wherea]
    #xb, yb, _ = ell_proj[whereb]
    theta = np.rad2deg(np.arctan(ya/xa))
    
    return a, b, theta


def foreshortening(theta, i):
    '''
    from de Pater et al 2006, doi:10.1016/j.icarus.2005.08.011
    corrects for foreshortening of inclined rings

    TO DO: accept astropy Angles, put into the ring modeling scripts

    i: degrees
    theta: radians
        (deal with it)
    '''
    B = np.pi / 2 - np.abs(np.deg2rad(i))
    return np.sqrt(np.sin(theta)**2 * np.sin(B)**2 + np.cos(theta)**2)


def ring_area(a, e, width, delta_width=0.0, B=90.0):
    '''
    Compute projected area of an eccentric ring with arbitrary opening angle
    and  from geometry
    
    Parameters
    ----------
    a: float or Quantity, required. semimajor axis [distance]
    e: float, required. eccentricity
    width: float or Quantity, required. average width of ring [distance]
        for an asymmetric ring, use (apoapsis_width + periapsis_width)/2
    delta_width: float or Quantity, optional. default 0.0. apoapsis width minus periapsis width [distance]
    B: float or Quantity, optional. default 90 (i.e., open). opening angle in degrees
    
    Returns: projected area in [distance unit]^2
    '''
    w_p = width - delta_width/2.
    w_a = width + delta_width/2.
    c = a*e
    a_i = a - (w_p + w_a)/4
    a_o = a + (w_p + w_a)/4
    c_i = c - (w_a - w_p)/4
    c_o = c + (w_a - w_p)/4
    e_i = c_i / a_i
    e_o = c_o / a_o
    A = np.pi * a_o**2 * np.sqrt(1 - e_o**2) - np.pi * a_i**2 * np.sqrt(1 - e_i**2)
    Ab = A * np.sin(np.radians(B))
    return Ab


class Ring:

    def __init__(self, a, e, omega, i, w, width=1.0, flux=1.0):
        '''
        model of a planetary ring

        Parameters
        ----------
        a : semimajor axis. assumes km if not an astropy Quantity
        e : eccentricity
        Omega : longitude of ascending node. assumes degrees if not an astropy Quantity
        i : inclination. assumes degrees if not an astropy Quantity
        w : argument of periapsis. assumes degrees if not an astropy Quantity
        width : float or Quantity, optional. default 1 km (i.e., very thin).
            assumes km if not an astropy Quantity
        flux : float or Quantity, optional. default 1.0.

        Attributes
        ----------
        a : semimajor axis
        e : eccentricity
        omega : longitude of ascending node
        i : inclination
        w : argument of periapsis
        width : ring width; semimajor axis a is at center.
        flux :


        Examples
        --------
        
        '''
        # to do: write tests that pass astropy Quantities with units other than
        # km and deg

        self.a = u.Quantity(a, unit=u.km)
        self.e = e
        self.b = b_from_ae(self.a, self.e)
        self.c = self.e * self.a
        self.omega = Angle(omega, u.deg)
        self.i = Angle(i, u.deg)
        self.w = Angle(w, u.deg)
        self.width = u.Quantity(width, unit=u.km)
        self.flux = flux

    def __str__(self):
        '''
        String representation

        Examples
        --------
        >>> from pylanetary.rings import Ring
        >>> epsilon_ring = Ring(whatever)
        >>> print(epsilon_ring)
        Ring instance; a=whatever, e=whatever, i=whatever, width=whatever
        '''
        return f'Ring instance; a={self.a}, e={self.e}, i={self.i}, omega={self.omega}, w={self.w}, width={self.width}'

    def as_orbit(self, T=1, tau=0):
        '''
        Is this a good idea to have velocities and stuff?
        Might want to re-think what functionality we really want here

        Parameters
        ----------
        T : orbital period
        tau : time of periapsis passage

        returns
        -------
        PyAstronomy Keplerian Ellipse object

        examples
        --------
        >>> epsilon_ring = Ring(a, e, omega, i, w)
        >>> orbit = epsilon_ring.as_orbit(T, tau=0)
        >>> print(orbit.pos)
        >>> print(orbit.radius)
        >>> print(orbit.vel)
        >>> print(orbit.peri)
        '''

        # decide: is it this code's job to calculate the orbital period
        # from semimajor axis based on planet mass?
        # would require planet masses in a data table
        # if so, can do later

        #from PyAstronomy import pyasl
        #ke = pyasl.KeplerEllipse(self.a, T, tau = self.tau, e = self.e, Omega = self.omega, i = self.i, w = self.w)
        # return ke
        return

    def as_elliptical_annulus(
            self,
            focus,
            pixscale,
            width=None,
            n=1e3,
            return_params=False):
        '''
        return elliptical annulus surrounding the ring of the given width
        in pixel space

        Parameters
        ----------
        focus : tuple, required. location of planet (one ellipse focus) in pixels
        pixscale : float or Quantity, required. assumes km if not an astropy Quantity
        width : true (non-projected) width of ring. astropy quantity required
        n: number of data points to rotate and project; higher n means more accurate estimation
            of projected a, b, theta
        return_params: bool, optional. default False. If True,
            also return dict with the model ellipse included

        To do:
            experiment with using manually-defined b_in to make epsilon-like ring
            make this return the periapsis and apoapsis angle
        '''

        # convert to simple floats instead of astropy unit quantities
        a, b = self.a.to(u.km).value, self.b.to(u.km).value
        omega, i, w = self.omega.to(u.deg).value, self.i.to(u.deg).value, self.w.to(u.deg).value
        if width is None:
            width = self.width
        width = width.to(u.km).value
        pixscale = u.Quantity(pixscale, unit=u.km)

        # rotate and project the ellipse
        true_params = project_ellipse(a, self.e, i, omega, w, n=int(n), origin=np.array([0, 0, 0]), proj_plane=[0, 0, 1])
        a_f, b_f, theta_f = calc_abtheta(true_params['ell'])
        a_f, b_f = np.abs(a_f), np.abs(b_f)

        # scale the width with the geometry
        a_outer = a_f + (a_f / a) * (width / 2)
        a_inner = a_f - (a_f / a) * (width / 2)
        b_outer = b_f + (b_f / b) * (width / 2)
        #b_inner = b_f - (b_f/self.b)*(width/2)

        # put center of image at one focus
        # remove extraneous zero in z dimension
        center = -true_params['f0'][:2]

        # convert to pixel values
        a_inner = a_inner / pixscale.value
        a_outer = a_outer / pixscale.value
        # account for inclination, which effectively shortens b even more
        b_outer = b_outer / pixscale.value
        center = np.array(focus) - center / pixscale.value
        center = center[::-1]

        # finally make the annulus object
        ann = aperture.EllipticalAnnulus(center,
                                         a_in=a_inner,
                                         a_out=a_outer,
                                         b_out=b_outer,
                                         b_in=None,
                                         theta=np.deg2rad(theta_f))

        if return_params:
            return ann, true_params
        return ann

    def as_azimuthal_wedges(
            self,
            shape,
            focus,
            pixscale,
            nwedges=60,
            width=None,
            n=1e4,
            z=5):
        '''
        return n partial elliptical annulus wedges

        Parameters
        ----------
        shape: tuple, required. shape of image in pixels
        focus: tuple, required. location of planet (one ellipse focus) in pixels
        pixscale: astropy Quantity or float in km
        nwedges: number of wedges to compute
        width: astropy quantity required
        n: number of points for as_elliptical_annulus to compute. see that docstring
            for details
        z: factor for ndimage zoom; larger makes more accurate wedge areas

        Returns
        -------
        theta_list: angle corresponding to lower corner of wedge
        ann_list: list of wedge masks

        Notes
        -----
        current implementation removes foreshortening correction "magically"
            by making the 

        To-do list
        ----------
        This is computationally expensive!
        a better implementation would be to make a photutils object
        for wedges of an ellipse. but this requires Cython programming
        '''

        # handle input params
        if width is None:
            width = self.width
        pixscale = u.Quantity(pixscale, unit=u.km)

        zshape = (shape[0] * z, shape[1] * z)
        zfocus = (focus[0] * z, focus[1] * z)
        ann = self.as_elliptical_annulus(
            zfocus, pixscale / z, width=width, n=n, return_params=False)
        # _, ringplane_params = self.as_elliptical_annulus(zfocus, pixscale/z,
        # width=width, n=nwedges, return_params=True) #this to get equal ring
        # plane azimuth angles of length (nwedges,)
        ann = ann.to_mask().to_image(zshape)
        width = z * width.to(u.km).value

        '''
        # use ellipse params to find ring-plane azimuth angles
        x = ringplane_params['ell'].T[0]
        y = ringplane_params['ell'].T[1]
        theta_list = np.arctan2(y,x)
        theta_list[theta_list < 0.0] += 2*np.pi
        theta_list = np.sort(theta_list)

        dtheta_list = theta_list[1:] - theta_list[:-1]
        last_element = 2*np.pi - (theta_list[-1] - theta_list[0]) # handle circle wrapping
        dtheta_list = np.concatenate([dtheta_list, np.array([last_element])])
        '''

        # image plane azimuth angles
        theta_list = np.linspace(0, 2 * np.pi - 2 * np.pi / nwedges, nwedges)
        dtheta_list = 2 * np.pi / nwedges * np.ones(theta_list.shape)

        # constants in every loop; rectangle width and height to make wedges
        h = np.max(zshape)  # just make it huge
        # this can be large, just make it double a reasonable width scaling
        w = 2 * z * (self.a * np.sin(np.max(dtheta_list)) / pixscale).value

        # iterate over angle
        ann_list = []
        for i, theta in enumerate(theta_list):  # radians assumed everywhere
            dtheta = dtheta_list[i]

            # make a wedge out of two rectangles.
            # first find center knowing that one corner has to be at zfocus
            d = 0.5 * np.sqrt(w**2 + h**2)  # distance from corner to center
            phi = np.arctan(w / h)  # angle from rectangle base to center
            center1 = zfocus + d * \
                np.array([np.cos(theta + phi), np.sin(theta + phi)])
            center2 = zfocus + d * \
                np.array([np.cos(theta + phi + dtheta), np.sin(theta + phi + dtheta)])

            rect1 = aperture.RectangularAperture(
                center1, w, h, theta + np.pi / 2).to_mask().to_image(zshape)
            rect2 = aperture.RectangularAperture(
                center2, w, h, theta + np.pi / 2 + dtheta).to_mask().to_image(zshape)
            wedge = rect1 - rect2
            wedge[wedge < 0] = 0.0

            wedge_ann = wedge * ann  # this is only an approximation; zooming image recommended

            '''
            # diagnostic plot
            if theta <0.5:

                fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize = (15,6))
                ax0.imshow(ann, origin = 'lower')
                ax1.imshow(wedge, origin = 'lower')
                ax2.imshow(wedge_ann, origin = 'lower')
                plt.show()
            '''

            wedge_out = ndimage.zoom(wedge_ann, 1.0 / z)
            ann_list.append(wedge_out)

        return theta_list, ann_list

    def as_2d_array(
            self,
            shape,
            pixscale,
            focus=None,
            width=None,
            flux=None,
            beamsize=None):
        '''
        return a 2-d array that looks like a mock observation
        optional smearing over Gaussian beam

        Parameters
        ----------
        shape : tuple, required. output image shape
        pixscale : float/int or astropy Quantity, required. pixel scale
            of the output image. If float/int (i.e. no units specified), then
            kilometers is assumed
        focus : tuple, optional. pixel location of planet around which ring orbits.
            if not specified, center of image is assumed
        width : float/int or astropy Quantity. If float/int (i.e. no units specified), then
            kilometers is assumed
        flux : float/int or astropy Quantity. technically not a flux, but a specific intensity!
            sets specific intensity of ring
            NEED TO DECIDE: what default units make sense here? - probably a surface brightness

        beamsize : float/int or 3-element array-like, optional.
            FWHM of Gaussian beam with which to convolve the observation
            units of fwhm are number of pixels.
            if array-like, has form (FWHM_X, FWHM_Y, POSITION_ANGLE)
            units of position angle are assumed degrees unless astropy Angle is passed
            if float/int, this is FWHM of assumed circular beam
            if no beamsize is specified, will make infinite-resolution

        Returns
        -------
        2-d numpy array

        Examples
        --------


        '''
        # to do: write a test that passes pixscale with units other than km

        if flux is None:
            flux = self.flux
        if width is None:
            width = self.width
        pixscale = u.Quantity(pixscale, u.km)

        # put the ring onto a 2-D array using EllipticalAnnulus
        if focus is None:
            focus = (shape[0] / 2.0, shape[1] / 2.0)

        ann = self.as_elliptical_annulus(focus, pixscale, width)
        arr_sharp = flux*ann.to_mask(method='exact').to_image(shape)
        # since flux is simple multiply by mask, its really a specific intensity
        # e.g. Jy/sr

        if beamsize is None:
            return arr_sharp
        else:
            return convolve_with_beam(arr_sharp, beamsize)


class RingSystemModelObservation:

    def __init__(self,
                 planet,
                 location=None,
                 horizons_loc=None,
                 epoch=None,
                 ringnames=None,
                 fluxes='default'):
        '''
        make a model of a ring system
        combines static data tables, originally from e.g. https://pds-rings.seti.org/uranus/uranus_rings_table.html
            with the Ring Node query tool
            and the

        Parameters
        ----------
        planet: str, required. one of Jupiter, Saturn, Uranus, Neptune
        epoch : `~astropy.time.Time` object, or str in format YYYY-MM-DD hh:mm, optional.
                If str is provided then UTC is assumed.
                If no epoch is provided, the current time is used.
        location : str, or array-like, or `~astropy.coordinates.EarthLocation`, optional
            If str, named observeratory supported by the ring node, e.g. JWST.
            If array-like, observer's location as a
            3-element array of Earth longitude, latitude, altitude
            that istantiates an
            `~astropy.coordinates.EarthLocation`.  Longitude and
            latitude should be anything that initializes an
            `~astropy.coordinates.Angle` object, and altitude should
            initialize an `~astropy.units.Quantity` object (with units
            of length).  If ``None``, then the geofocus is used.
        horizons_loc: str, required
                JPL Horizons ephemeris tool observer location.
                Should match the other location parameter.
                TO DO LATER: lookup table to
                make only one location specification required
                e.g., '500@-170' for JWST
        ringnames : list, optional. which rings to include in the model
            if no ringnames provided then all rings are assumed.
            Case-sensitive! Typically capitalized, e.g. "Alpha"
                (for now - annoying to make case-insensitive)
        fluxes : list-like, optional. surface brightness units are expected, e.g. brightness temperature
            if fluxes == 'default', the optical depths are read in from the static table
                and exponentiated to more closely resemble surface brightness units
                so the result is the attenuation, ATT = 1 - exp(ringtable['Optical Depth'])
                assuming the emittance is small (violated for thermal observations, obviously)


        Attributes
        ----------
        planetname : str, name of planet
        rings : dict of ringmodel.Ring objects, with ring names as keys
                note ring names are case-sensitive! Typically capitalized, e.g. "Alpha"
        ringtable : table of ephemeris data as well as time-invariant parameters for
        systemtable :
        bodytable :
        np_ang :


        Examples
        --------
        Need an example of how to add a custom ring
        should be possible by just adding a ringmodel.Ring() object
            into the dict self.ring

        Need an example of how to modify ring data
        should be possible by just changing the ringmodel.Ring() object
            in the dict self.ring

        To-do list
        ----------
        Right now we are not accounting for the peculiar inclinations and
            arguments of periapsis of individual rings relative to the system
            but they are being passed into this code
            just need to write more geometry
        '''

        planet = planet.lower().capitalize()
        self.planetname = planet

        # fix inputs
        if epoch is None:
            raise NotImplementedError(
                "not done yet; please provide epoch for now")
        if location is None:
            raise NotImplementedError(
                "not done yet; please provide location for now")

        # send query to Planetary Ring Node, and query static data table
        node = RingNode()
        self.bodytable, self.ringtable = node.ephemeris(
            planet=planet, epoch=epoch, location=location, cache=False)
        self.systemtable = self.bodytable.meta

        ring_data_source = importlib.resources.open_binary(
            'pylanetary.rings.data', f'{planet}_ring_data.hdf5')
        #ring_static_data = table.Table.read(f'data/{planet}_ring_data.hdf5', format = 'hdf5')
        ring_static_data = table.Table.read(ring_data_source, format='hdf5')
        planet_ephem = self.bodytable.loc[planet]
        #self.ob_lat, self.ob_lon = planet_ephem['sub_obs_lat'], planet_ephem['sub_obs_lon']

        # query Horizons for the north polar angle
        obj = Horizons(id=horizons_lookup[planet.lower().capitalize()], location=horizons_loc, epochs={
                       'start': epoch.to_value('iso'), 'stop': (epoch + 1 * u.day).to_value('iso'), 'step': '1d'})
        eph = obj.ephemerides()
        self.eph = eph
        self.np_ang = eph['NPole_ang'][0]

        # match the static and ephemeris data for rings using a table merge
        ring_static_data.rename_column('Feature', 'ring')
        if self.ringtable is not None:
            self.ringtable = table.join(
                self.ringtable,
                ring_static_data,
                keys='ring',
                join_type='right')
        else:
            self.ringtable = ring_static_data
        self.ringtable.add_index('ring')

        if ringnames is None:
            ringnames = list(self.ringtable['ring'])

        # make self.ringtable and fluxes contain only the rings in ringnames
        self.ringtable = self.ringtable.loc[ringnames]
        if fluxes == 'default':
            fluxes = list(self.ringtable['Optical Depth'])
            # optical depth to 1 - transmittance
            fluxes = [1 - np.exp(-val) for val in fluxes]

        self.rings = {}
        for i in range(len(ringnames)):
            ringname = ringnames[i]
            flux = fluxes[i]
            try:
                ringparams = self.ringtable.loc[ringname]
            except Exception as e:
                raise ValueError(
                    f"Ring name {ringname} not found in the data table of known rings")

            # make a Ring object for each one
            if 'ascending node' in self.ringtable.loc['ring', ringname].keys():
                # + self.systemtable['sub_obs_lon']
                
                ## TO DO: FIX HERE ###
                ## the omega values in the Planetary Ring node correspond to the additional
                ## inclination values given in the static table.
                ## so we need to do a second rotation after the first one, 
                ## with the small inclinations
                ## given in the static table
                ## and this omega. 
                ## for now, just force omega to be the system omega
                
                #omega = self.np_ang * u.deg + \
                #    self.ringtable.loc['ring', ringname]['ascending node'].filled(0.0)
                omega = self.np_ang * u.deg
            else:
                omega = self.np_ang * u.deg
            if 'pericenter' in self.ringtable.loc['ring', ringname].keys():
                # filled() just turns from a masked array, which doesn't pass,
                # to an unmasked array. the fill value is not used
                w = self.ringtable.loc['ring',
                                       ringname]['pericenter'].filled(0.0)
            else:
                w = 0.0 * u.deg
            i = 90 * u.deg + self.systemtable['opening_angle']

            # many of the less-well-observed rings have masked values
            # for many of these quantities, particularly omega, i, w, or even e. these go to
            # zero when made into floats, so it is ok

            # handle the fact that half of the static ring tables have middle boundary and width,
            # while the other half have inner and outer boundary
            if 'Middle Boundary (km)' in ringparams.keys():
                a = ringparams['Middle Boundary (km)'] * u.km
                width = ringparams['Width']
            elif 'Inner Boundary (km)' in ringparams.keys():
                a = 0.5 * (ringparams['Inner Boundary (km)'] +
                           ringparams['Outer Boundary (km)']) * u.km
                width = ringparams['Outer Boundary (km)'] - \
                    ringparams['Inner Boundary (km)']
            else:
                raise ValueError(
                    'Neither "Middle Boundary (km)" nor "Inner Boundary (km)" found in static ring data')

            # handle eccentricity
            if 'Eccentricity' in ringparams.keys():
                e = ringparams['Eccentricity']
            else:
                e = 0.0
            thisring = Ring(a,
                            e,
                            omega,
                            i,
                            w,
                            width=width,
                            flux=flux)
            self.rings[ringname] = thisring

    def as_2d_array(self, shape, pixscale, focus=None, beamsize=None):
        '''
        return a 2-d array that looks like a mock observation
        optional smearing over Gaussian beam

        Parameters
        ----------
        shape : tuple, required. output image shape in number of pixels
        pixscale : float/int or astropy Quantity, required. pixel scale
            of the output image. If float/int (i.e. no units specified), then
            kilometers is assumed
        beamsize : float/int or 3-element array-like, optional.
            FWHM of Gaussian beam with which to convolve the observation
            units of fwhm are number of pixels.
            if array-like, has form (FWHM_X, FWHM_Y, POSITION_ANGLE)
            units of position angle are assumed degrees unless astropy Angle is passed
            if no beamsize is specified, will make infinite-resolution

        Returns
        -------
        2-d numpy array

        Examples
        --------


        '''

        arr_out = np.zeros(shape)
        for ringname in self.rings.keys():

            arr_out += self.rings[ringname].as_2d_array(
                shape, pixscale, focus=focus, beamsize=None)

        # run convolution with beam outside loop so it is only done once
        if beamsize is None:
            return arr_out
        else:
            return convolve_with_beam(arr_out, beamsize)
