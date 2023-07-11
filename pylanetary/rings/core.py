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

from ..utils import *


'''
To implement
------------
* make as_azimuthal_wedges() allow non-square images!
* class that inherits from Ring for asymmetric rings like Uranus's epsilon ring
* make azimuthal wedges code much faster by implementing wedges as a photutils object
'''

horizons_lookup = {
    'Mars': '499',
    'Jupiter': '599',
    'Saturn': '699',
    'Uranus': '799',
    'Neptune': '899',
    'Pluto': '999'}


def vector_magnitude(a):
    '''
    this is a bit faster than using np.linalg.norm
    
    Parameters
    ----------
    :a: np.array, required, shape (n, 3) or (n, 2)
        corresponding to n vectors
    
    Returns
    -------
    np.array, same shape as input, magnitude of input vector a
    '''
    a = np.array(a)
    if len(a.shape) == 1:
        return np.sqrt(np.sum(a * a))
    return np.sqrt((a * a).sum(axis=1))
    
    
def vector_normalize(v):
    '''
    returns the original vector if it has length zero
    
    Parameters
    ----------
    :v: np.array, required, shape (n, 3) or (n, 2)
        corresponding to n vectors
    
    Returns
    -------
    np.array, same shape as input. the normalized vectors
    '''
    norm = vector_magnitude(v)
    if norm.size > 1:
        norm[norm == 0] = 1.0 #hack to handle zero-length vectors
        return v / norm[:,np.newaxis]
    elif norm.size == 1:
        if norm == 0:
            return v
        else:
            return v/norm


def plane_project(X, Z):
    '''
    projects vectors X into plane normal to vector Z
    
    Parameters
    ----------
    :X: np.array, required, shape (n,3). vectorss to project
    :Z: np.array, required, shape (3,). direction to project in.
    
    Returns
    -------
    np.array, shape (3,). the projected vector.
    '''
    zhat = vector_normalize(Z)
    return np.cross(zhat, np.cross(X, zhat))


def double_rotate(vec, params_ring, params_sys):
    '''
    rotate system * rotate ring
    same convention as planetary ring node
    to agree with planetary ring node, params_sys = [90, B - 90, np_ang]
    and params_ring = [w, i, 0]
    
    Parameters
    ----------
    :vec: np.array, required, shape (n, 3). 
        3-element [x, y, 0] vectors on the perimeter of the ring 
        i.e. semimajor axis is [a, 0, 0] and semiminor axis is [0, b, 0]
    :params_ring: np.array, required.
        params_ring = [w, i, omega] all in degrees w.r.t the ring plane
    :params_sys: np.array, required. 
        params_sys = [w, i, omega] in degrees for the ring plane w.r.t. observer
    
    Returns
    -------
    np.array, shape (n, 3), the rotated vectors
    '''
    rot_ring = Rotation.from_euler('zxz', params_ring, degrees=True)
    rot_sys = Rotation.from_euler('zxz', params_sys, degrees=True)
    rot_total = rot_sys * rot_ring
    vec_out = rot_total.apply(vec)
    return vec_out


def b_from_ae(a, e):
    '''
    Parameters
    ----------
    :a: float, required.
        semimajor axis of ellipse
    :e: float, required.
        eccentricity of ellipse
    
    Returns
    -------
    float, semiminor axis of ellipse
    '''
    return a * np.sqrt(1 - e**2)


def vector_ellipse(u, v, n, origin=np.array([0, 0, 0])):
    '''
    Vector equation of an ellipse in two dimensions
    
    Parameters
    ----------
    :u: np.array, required, shape (3,).
        vectorized semimajor axis, e.g. [a,0,0]
    :v: np.array, required, shape (3,)
        vectorized semiminor axis, e.g. [0,b,0]
        should be perpendicular to u
    :n: int, required.
        number of vectors to compute
    :origin: np.array, required, shape (3,)
        vector pointing to the center of the ellipse
    
    Returns
    -------
    np.array, required, shape (n, 3)
        vectors pointing to the circumference of the ellipse
        
    
    References
    ----------
    https://math.stackexchange.com/questions/3994666/parametric-equation-of-an-ellipse-in-the-3d-space#:~:text=In%20the%20parametric%20equation%20x,a%20point%20with%20minimum%20curvature.
    '''
    t = np.linspace(0, 2*np.pi, n+1)[:-1]
    u = u[np.newaxis, :]
    v = v[np.newaxis, :]
    t = t[:, np.newaxis]

    return origin + u * np.cos(t) + v * np.sin(t)
    

def project_ellipse_double(a, e, params_ring, params_sys, origin = np.array([0,0,0]), n=50, proj_plane = [0,0,1]):
    '''
    Description
    -----------
    make a projection of an ellipse representing a ring
    with ring system parameters as Euler angles [w, i, omega]
    and individual ring parameters relative to the system 
        also as Euler angles [w, i, omega]
    
    The angles params=[w, i, omega] and params_sys=[w_sys, i_sys, omega_sys] 
    represent two sets of Euler angles [gamma, beta, alpha]
    such that the total rotation is given by
    (params_sys rotation matrix) * (params rotation matrix)
    
    This code builds an ellipse in the x,y plane represented by vectors, 
    applies the double rotation to those vectors, then re-projects the
    ellipse into the x,y plane. This simulates an observation of a ring. 
    
    References
    ~~~~~~~~~~
    https://en.wikipedia.org/wiki/Euler_angles
    https://en.wikipedia.org/wiki/Orbital_elements#Euler_angle_transformations
    
    Parameters
    ----------
    :a: float, required.
        semimajor axis in any distance unit.
    :e: float, required.
        eccentricity, unitless.
    :params_ring: np.array, required, shape (3,).
        [w, i, omega] of ring relative to ring system.
    :params_sys: np.array, required, shape (3,).
        [w, i, omega] of ring plane relative to observer.  
    :n: int, required.
        number of equally-spaced points in ellipse circumference to compute
    :origin: np.array, required, shape (3,).
        vector pointing to center of the ellipse
    :proj_plane: np.array, required, shape (3,).
        normal vector to plane that you want to re-project back to after rotation
        typically this will be [0, 0, 1], i.e., re-project to x,y plane
    
    Returns
    -------
    dictionary of projected ellipse parameters with the following keys:
        a: np.array, shape (3,), projected semimajor axis of ring
        b: np.array, shape (3,), projected semiminor axis of ring
        f0: np.array, shape (3,), vector to one focus
        f1: np.array, shape (3,), vector to the other focus
        ell: np.array, shape (n,3), vectors along circumference of ring
    
    '''
    b = a*np.sqrt(1-e**2)
    c = a*e
    f0 = np.array([origin[0] + c, origin[1], 0]) # foci
    f1 = np.array([origin[0] - c, origin[1], 0])
    a_vec = np.array([a,0,0])
    b_vec = np.array([0,b,0])
    
    # apply double rotation to a, b, f0, f1
    a_rot = double_rotate(a_vec, params_ring, params_sys)
    b_rot = double_rotate(b_vec, params_ring, params_sys)
    f0_rot = double_rotate(f0, params_ring, params_sys)
    f1_rot = double_rotate(f1, params_ring, params_sys)
    
    # project these
    a_proj, b_proj = plane_project(a_rot, proj_plane), plane_project(b_rot, proj_plane)
    f0_proj = plane_project(f0_rot, proj_plane)
    f1_proj = plane_project(f1_rot, proj_plane)
    
    # make and project ellipse circumference
    ell = vector_ellipse(a_vec, b_vec, n, origin=origin)
    ell_rot = double_rotate(ell, params_ring, params_sys)
    ell_p = plane_project(ell_rot, proj_plane)
    
    # dict of outputs
    output = {'a':a_proj,
             'b':b_proj,
             'f0':f0_proj,
             'f1':f1_proj,
             'ell':ell_p}
    
    return output


def calc_abtheta(ell):
    '''
    given vectors defining the circumference of an ellipse, 
    find corresponding values of a, b, and theta
    using the fact that locations of a, b are max, min of ellipse vectors
    
    Parameters
    ----------
    :ell: np.array, required, shape (n, 3), vectors along circumference of ring
        
    Returns
    -------
    :a: float, semimajor axis of ellipse
    :b: float, semiminor axis of ellipse
    :theta: float, rotation angle of ellipse in degrees
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
    corrects for foreshortening of inclined rings
    
    Parameters
    ----------
    :theta: float, required.
        units radians. azimuth angle along the ring from 0 to 2*pi
    :i: float, required.
        units degrees. inclination of ring

    Returns
    -------
    Float, fraction by which to multiply ring brightness at azimuth
        angle theta to correct foreshortening.
    
    References
    ----------
    de Pater et al 2006, doi:10.1016/j.icarus.2005.08.011
    
    To Do
    -----
    needs to accept Astropy angles
    '''
    B = np.pi / 2 - np.abs(np.deg2rad(i))
    return np.sqrt(np.sin(theta)**2 * np.sin(B)**2 + np.cos(theta)**2)


def ring_area(a, e, width, delta_width=0.0, B=90.0):
    '''
    Compute projected area of an eccentric ring at a given opening angle
    and peri-apo width asymmetry
    
    Parameters
    ----------
    :a: float, required. 
        [distance] semimajor axis 
    :e: float, required. 
        eccentricity
    :width: float, required.
        [distance] average width of ring 
        for an asymmetric ring, use (apoapsis_width + periapsis_width)/2
    :delta_width: float, optional. default 0.0
        [distance] apoapsis width minus periapsis width
    :B: float, optional. default 90 (i.e., open)
        [degrees] ring opening angle
    
    Returns
    -------
    projected area in [distance unit]^2
    
    References
    ----------
    Molter et al. 2019, doi:10.3847/1538-3881/ab258c
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
    '''
    model of a planetary ring
    '''

    def __init__(self, a, e, w, i, omega, width=1.0, flux=1.0, params_sys=[0.0, 0.0, 0.0]):
        '''
        Parameters
        ----------
        :a: float or Quantity, required.
            semimajor axis. assumes km if not an astropy Quantity
        :e: float, required.
            eccentricity
        w: float or Quantity, required.
            argument of periapsis. assumes degrees if not an astropy Quantity
        i: float or Quantity, required.
            inclination. assumes degrees if not an astropy Quantity
        :omega: float or Quantity, required.
            longitude of ascending node. assumes degrees if not an astropy Quantity
        :width: float or Quantity, optional. default 1 km (i.e., very thin).
            full width of ring. assumes km if not an astropy Quantity
        :flux: float or Quantity, optional. default 1.0.
            ring flux density
        :params_sys: np.array, optional, shape (3,)
            [w, i, omega] of ring system plane relative to observer

        Attributes
        ----------
        same as parameters

        Examples
        --------
        
        To Do
        -----
        decide on default units of flux density
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
        self.params_sys = params_sys

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
        raise NotImplementedError
        return

    def as_elliptical_annulus(
            self,
            focus,
            pixscale,
            width=None,
            n=1e3,
            return_params=False):
        '''
        make Astropy aperture photometry elliptical annulus object
        surrounding the ring

        Parameters
        ----------
        :focus: tuple, required. 
            location of planet (one of the foci of the ellipse) in pixels
        :pixscale: float or Quantity, required. 
            pixel scale of observations. assumes km/px if not Quantity
        :width: float or Quantity, required.
            true (non-projected) full width of ring. assumes km if not Quantity
        :n: int, optional. default 1000.
            number of data points to rotate and project
            higher n means more accurate estimation
            of aperture.EllipticalAnnulus projected a, b, theta values
        return_params: bool, optional. default False. 
            If True, return model ellipse dict
            see docstring of project_ellipse_double()

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
        true_params = project_ellipse_double(
                            a, 
                            self.e, 
                            np.array([w, i, omega]), 
                            n=int(n), 
                            origin=np.array([0, 0, 0]), 
                            proj_plane=[0, 0, 1], 
                            params_sys=self.params_sys)
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
        b_outer = b_outer / pixscale.value
        center = np.array(focus) - center / pixscale.value
        #center = center[::-1]

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
            pixscale,
            focus=None,
            nwedges=60,
            width=None,
            n=1e3,
            z=5):
        '''
        return n partial elliptical annulus wedges, e.g. for computing
        azimuthal profile of ring

        Parameters
        ----------
        :shape: tuple, required. 
            shape of image in pixels
        :focus: tuple, required. 
            location of planet (one ellipse focus) in pixels
        :pixscale: astropy Quantity or float, required
            pixel scale of image. assumes km/px if not Quantity
        :nwedges: int, optional. default 60.
            number of wedges to compute
        :width: astropy quantity required
        :n: int, optional. default 1000.
            number of points for as_elliptical_annulus to compute. 
            see that docstring for details.
            should be much larger than nwedges
        z: int, optional. default 5
            factor for ndimage.zoom
            larger makes more accurate wedge areas

        Returns
        -------
        np.array, shape (nwedges,)
            angles in radians from 0 to 2*pi corresponding to lower corner of wedge
        list, len nwedges. list of wedge masks

        Notes
        -----
        * current implementation removes foreshortening correction "magically"
            by making the angular width of the wedges in the image plane
            see ring-system-modeling-tutorial.ipynb for a more detailed
            explanation and example

        To-do
        -----
        This is computationally expensive!
        a better implementation would be to make a photutils object
        for wedges of an ellipse. but this requires Cython and
        a lot of geometry
        '''

        # handle input params
        if width is None:
            width = self.width
        if focus is None:
            focus = (shape[0] / 2.0, shape[1] / 2.0)
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

            wedge_out = rebin(wedge_ann, 1.0 / z)
            ann_list.append(wedge_out)

        return theta_list, ann_list


    def as_2d_array(
            self,
            shape,
            pixscale,
            focus=None,
            width=None,
            flux=None,
            beam=None):
        '''
        make a 2-d array that looks like a mock observation of the ring
        with optional smearing over a beam

        Parameters
        ----------
        :shape: tuple, required. 
            output image shape
        pixscale: float or astropy Quantity, required. 
            pixel scale of the output image. assumes km/px if not Quantity
        :focus: tuple, optional, default None.
            pixel location of planet around which ring orbits.
            if not specified, center of image is assumed
        :width: float or Quantity, optional. default None.
            non-projected (face-on) full width of ring.
            if not specified, assumes width of 1 km.
        :flux: float or Quantity, optional, default None.
            technically not a flux, but a specific intensity!
            sets specific intensity of ring
            if not specified, assumes 1 [unit??].
        :beam: float, 3-element array-like, or 2-d array, optional, default None.
            Gaussian beam with which to convolve the observation
            see docstring of utils.convolve_with_beam()

        Returns
        -------
        np.array, beam-convolved ring image

        Examples
        --------

        '''
        if flux is None:
            flux = self.flux
        if width is None:
            width = self.width
        pixscale = u.Quantity(pixscale, u.km)

        # put the ring onto a 2-D array using EllipticalAnnulus
        if focus is None:
            focus = (shape[0] / 2.0, shape[1] / 2.0)

        ann, params_out = self.as_elliptical_annulus(focus, pixscale, width, return_params=True)
        arr_sharp = flux*ann.to_mask(method='exact').to_image(shape)
        # since flux is simple multiply by mask, its really a specific intensity
        # e.g. Jy/sr

        if beamsize is None:
            return arr_sharp
        else:
            return convolve_with_beam(arr_sharp, beamsize)


class RingSystemModelObservation:
    '''
    model ring system combining static data from
    https://pds-rings.seti.org/uranus/uranus_rings_table.html
    with Planetary Ring Node ephemeris
    and JPL Horizons ephemeris
    '''

    def __init__(self,
            planet,
            location=None,
            horizons_loc=None,
            epoch=None,
            ringnames=None,
            fluxes='default'):
        '''
        Parameters
        ----------
        :planet: str, required. 
            one of Jupiter, Saturn, Uranus, Neptune
        :epoch: `~astropy.time.Time` object or str, optional. default None.
            if str, should be given in format YYYY-MM-DD hh:mm (assumes UTC)
            if None, the current time is used.
        :location: str, array-like, or `~astropy.coordinates.EarthLocation`, optional.
            If str, named observeratory supported by the ring node, e.g. JWST.
            If array-like, observer's location as a
            3-element array of Earth longitude, latitude, altitude
            that istantiates an `~astropy.coordinates.EarthLocation` object. 
            Longitude and latitude should be anything that initializes an
            `~astropy.coordinates.Angle` object, and altitude should
            initialize an `~astropy.units.Quantity` object (with units
            of length).  
            If ``None``, then the geofocus is used.
        :horizons_loc: str, required.
            JPL Horizons observer location code.
            Should match the other location parameter.
            TO DO LATER: lookup table to
            make only one location specification required
            e.g., '500@-170' for JWST
        :ringnames: list, optional. 
            names of rings to include in the model
            if no ringnames provided then all rings are assumed.
            Case-sensitive! Typically capitalized, e.g. "Alpha"
        :fluxes: list, np.array, or "default", optional. default "default".
            brightnesses associated with rings specified in ringnames.
            surface brightness units are expected, e.g. brightness temperature
            if fluxes == 'default', optical depths are read in from the static table
            and exponentiated to more closely resemble surface brightness units
            so the result is the attenuation, ATT = 1 - exp(ringtable['Optical Depth'])
            assuming the emittance is small (violated for thermal observations, obviously)

        Attributes
        ----------
        :planetname: str
            name of planet
        :rings: dict of ringmodel.Ring objects, with ring names as keys
            note ring names are case-sensitive! Typically capitalized, e.g. "Alpha"
        :ringtable: Astropy table of ephemeris data from ring node query tool
            as well as time-invariant parameters for each ring from static table
        :systemtable: Astropy table of ephemeris data for ring system from ring node query tool
        :bodytable: Astropy table of ephemeris data for satellites from ring node query tool
        :ephem: JPL Horizons ephemeris
        :np_ang: north polar angle


        Examples
        --------
        Need an example of how to add a custom ring
        should be possible by just adding a ringmodel.Ring() object
            into the dict self.ring

        Need an example of how to modify ring data
        should be possible by just changing the ringmodel.Ring() object
            in the dict self.ring

        To Do
        -----
        * Not yet understood why omega must equal 0 for individual rings
            relative to ring system in order to make argument of periapsis agree
            with the Planetary Ring Node.
        * Make this work nicely with utils.Body object
        * implement default epoch and location
        * lookup table to avoid needing to specify both location and horizons_loc
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

        # query the Planetary Ring Node
        node = RingNode()
        self.bodytable, self.ringtable = node.ephemeris(
            planet=planet, epoch=epoch, location=location, cache=False)
        self.systemtable = self.bodytable.meta
        
        # query static data table
        ring_data_source = importlib.resources.open_binary(
            'pylanetary.rings.data', f'{planet}_ring_data.hdf5')
        ring_static_data = table.Table.read(ring_data_source, format='hdf5')
        planet_ephem = self.bodytable.loc[planet]
        #self.ob_lat, self.ob_lon = planet_ephem['sub_obs_lat'], planet_ephem['sub_obs_lon']

        # query Horizons for the north polar angle
        obj = Horizons(id=horizons_lookup[planet.lower().capitalize()], location=horizons_loc, epochs={
                       'start': epoch.to_value('iso'), 'stop': (epoch + 1 * u.day).to_value('iso'), 'step': '1d'})
        ephem = obj.ephemerides()
        self.ephem = ephem
        self.np_ang = ephem['NPole_ang'][0]

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
        
        # compute fluxes from optical depth (very approximate)
        if fluxes == 'default':
            taus = np.array(self.ringtable['Optical Depth'])
            # optical depth to 1 - transmittance
            taus[np.isnan(taus)] = 1.0 # lazy fix of bad table data, re-think this later
            fluxes = 1 - np.exp(-taus)
            

        # instantiate ring objects for all the rings
        params_sys = [90.0, self.systemtable['opening_angle'].value - 90.0, self.np_ang] #[w, i, omega]
        self.rings = {}
        for i in range(len(ringnames)):
            ringname = ringnames[i]
            flux = fluxes[i]
            try:
                ringparams = self.ringtable.loc[ringname]
            except Exception as e:
                raise ValueError(
                    f"Ring name {ringname} not found in the data table of known rings")

            e = 0.0
            i = 0.0
            omega = 0.0
            w = 0.0
            if 'Eccentricity' in ringparams.keys():
                e = ringparams['Eccentricity']
            if 'inclination' in ringparams.keys():
                i = ringparams['inclination'].filled(0.0)
            if 'ascending node' in ringparams.keys():
                omega = ringparams['ascending node'].filled(0.0)
            if 'pericenter' in ringparams.keys():
                w = ringparams['pericenter'].filled(0.0)
            # filled() just turns from a masked array, which doesn't pass,
            # to an unmasked array. the fill value is not used
            # many of the less-well-observed rings have masked values
            # for many of these quantities, particularly omega, i, w
            # these go to zero when made into floats, 
            # so the filled() hack is ok   

            # find semimajor axis
            # handling the fact that half the static ring tables have middle boundary and width,
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

            omega = 0.0 #this is required to get argument of periapsis correct, but I don't know why
            thisring = Ring(a,
                            e,
                            w,
                            i,
                            omega,
                            width=width,
                            flux=flux,
                            params_sys=params_sys)
            self.rings[ringname] = thisring

    def as_2d_array(self, shape, pixscale, focus=None, beam=None):
        '''
        return a 2-d array that looks like a mock observation
        optional smearing over Gaussian beam

        Parameters
        ----------
        :shape: tuple, required. 
            output image shape in number of pixels
        :pixscale: float or Quantity, required. 
            pixel scale of the output image. if not Quantity, assumes units of km/px
        :focus: tuple, optional.
            pixel location of planet center
            if None, assumes center of image.
        :beam: float, 3-element array-like, or 2-d np.array, optional.
            Gaussian beam with which to convolve the observation
            see docstring of utils.convolve_with_beam()
            if no beamsize is specified, will make infinite-resolution

        Returns
        -------
        np.array, mock observation image
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
