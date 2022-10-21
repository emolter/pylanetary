
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



'''
goal: 
    pull together JPL Horizons query, ring node query, 
    and static ring data from data/ to make model ring
    systems as observed from any location
    
    is there a better/more astropy-like library to import for keplerian ellipses than PyAstronomy?
        yes, but it is annoying to use
        much later: get an undergrad to make Keplerian ellipse module of Astropy
'''

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
        return np.sum(np.sqrt(a*a))
    return np.sqrt((a*a).sum(axis=1))
    

def plane_project(x,z):
    '''projects x onto plane where z is normal vector to that plane'''
    z = vector_normalize(z)
    return np.cross(z, np.cross(x, z))
    

def make_rot(i,omega,w):
    '''use i, omega, and w as Euler rotation angles and return a rotation object'''
    return Rotation.from_euler('zxz', [w, i, omega], degrees=True)
    

def rotate_and_project(vec,rot,proj_plane=[0,0,1]):
    '''
    use i, omega, and w as Euler rotation angles to project a vector
    first into 3-D then onto a 2-D plane
    '''
    return plane_project(rot.apply(vec), proj_plane)
    

def b_from_ae(a,e):
    return a*np.sqrt(1 - e**2)
    
    
def vector_ellipse(u, v, t, origin=np.array([0,0,0])):
    '''
    https://math.stackexchange.com/questions/3994666/parametric-equation-of-an-ellipse-in-the-3d-space#:~:text=In%20the%20parametric%20equation%20x,a%20point%20with%20minimum%20curvature.
    u, v are vectors corresponding to the vectorized a, b
    t are the data points from 0 to 2pi
    '''
    u = u[np.newaxis,:]
    v = v[np.newaxis,:]
    t = t[:,np.newaxis]
    #print(u*np.cos(t))
    
    return origin + u*np.cos(t) + v*np.sin(t)


def project_ellipse(a,e,i,omega,w,n=1000, origin = np.array([0,0,0]), proj_plane = [0,0,1]):
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
    t = np.linspace(0,2*np.pi,int(n))
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
    B = np.pi/2 - np.abs(np.deg2rad(i))
    return np.sqrt( np.sin(theta)**2 * np.sin(B)**2 + np.cos(theta)**2 )


class Ring:
    
    def __init__(self, a, e, omega, i, w, width = 1.0, flux = 1.0):
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
        # to do: write tests that pass astropy Quantities with units other than km and deg
        
        self.a = u.Quantity(a, unit=u.km)
        self.e = e
        self.b = b_from_ae(self.a, self.e)
        self.c = self.e*self.a
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
        #return ke
        return
    
            
    def as_elliptical_annulus(self, focus, pixscale, width=None, n=1e3, return_params = False):
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
        '''
        
        # convert to simple floats instead of astropy unit quantities
        a, b = self.a.to(u.km).value, self.b.to(u.km).value
        omega, i, w = self.omega.to(u.deg).value, self.i.to(u.deg).value, self.w.to(u.deg).value
        if width is None:
            width = self.width
        width = width.to(u.km).value
        pixscale = u.Quantity(pixscale, unit=u.km)
        
        # rotate and project the ellipse
        true_params = project_ellipse(a,self.e,i,omega,w,n=int(n), origin = np.array([0,0,0]), proj_plane = [0,0,1])
        a_f,b_f,theta_f = calc_abtheta(true_params['ell'])
        a_f, b_f = np.abs(a_f), np.abs(b_f)
        
        # scale the width with the geometry
        # test this!
        a_outer = a_f + (a_f/a)*(width/2)
        a_inner = a_f - (a_f/a)*(width/2)
        b_outer = b_f + (b_f/b)*(width/2)
        #b_inner = b_f - (b_f/self.b)*(width/2)
        
        # put center of image at one focus
        center = -true_params['f0'][:2] #remove extraneous zero in z dimension

        # convert to pixel values
        a_inner = a_inner/pixscale.value
        a_outer = a_outer/pixscale.value
        b_outer = b_outer/pixscale.value # account for inclination, which effectively shortens b even more
        center = np.array(focus) - center/pixscale.value

        # finally make the annulus object
        ann = aperture.EllipticalAnnulus(center, 
                            a_in=a_inner, 
                            a_out=a_outer, 
                            b_out=b_outer, 
                            b_in= None, 
                            theta = np.deg2rad(theta_f))
        
        if return_params:
            return ann, true_params
        return ann
    
        
    def as_azimuthal_wedges(self, shape, focus, pixscale, nwedges=60, width=None, n=1e4, z=5):
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
        current implementation makes them equal in angle in projected space
        but should really be equal in real azimuth... to fix!
        perfect data experiments would be good to check if 
        making equal azimuth wedges removes foreshortening correction
        
        This is computationally expensive!
        a better implementation would be to make a photutils object
        for wedges of an ellipse. but this is hard to do
        '''
        
        # handle input params
        if width is None:
            width = self.width
        pixscale = u.Quantity(pixscale, unit=u.km)
        
        zshape = (shape[0]*z, shape[1]*z)
        zfocus = (focus[0]*z, focus[1]*z)
        ann = self.as_elliptical_annulus(zfocus, pixscale/z, width=width, n=n, return_params=False)
        #_, ringplane_params = self.as_elliptical_annulus(zfocus, pixscale/z, width=width, n=nwedges, return_params=True) #this to get equal ring plane azimuth angles of length (nwedges,)
        ann = ann.to_mask().to_image(zshape)
        width = z*width.to(u.km).value
        
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
        theta_list = np.linspace(0, 2*np.pi - 2*np.pi/nwedges, nwedges)
        dtheta_list = 2*np.pi / nwedges * np.ones(theta_list.shape)
        
        # constants in every loop; rectangle width and height to make wedges
        h = np.max(zshape) #just make it huge
        w = 2*z*(self.a * np.sin(np.max(dtheta_list))/pixscale).value #this can be large, just make it double a reasonable width scaling
        
        # iterate over angle
        ann_list = []
        for i, theta in enumerate(theta_list): #radians assumed everywhere
            dtheta = dtheta_list[i]
            
            # make a wedge out of two rectangles.
            # first find center knowing that one corner has to be at zfocus
            d = 0.5*np.sqrt(w**2 + h**2) #distance from corner to center
            phi = np.arctan(w/h) #angle from rectangle base to center
            center1 = zfocus + d*np.array([np.cos(theta + phi), np.sin(theta + phi)])
            center2 = zfocus + d*np.array([np.cos(theta + phi + dtheta), np.sin(theta + phi + dtheta)])
            
            rect1 = aperture.RectangularAperture(center1, w, h, theta+np.pi/2).to_mask().to_image(zshape)
            rect2 = aperture.RectangularAperture(center2, w, h, theta+np.pi/2 + dtheta).to_mask().to_image(zshape)
            wedge = rect1 - rect2
            wedge[wedge<0] = 0.0
            
            wedge_ann = wedge * ann # this is only an approximation; zooming image recommended
            
            '''
            # diagnostic plot
            if theta <0.5:
                
                fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize = (15,6))
                ax0.imshow(ann, origin = 'lower')
                ax1.imshow(wedge, origin = 'lower')
                ax2.imshow(wedge_ann, origin = 'lower')
                plt.show()
            '''
            
            wedge_out = ndimage.zoom(wedge_ann, 1.0/z)
            ann_list.append(wedge_out)
            
        return theta_list, ann_list
    
        
    def as_2d_array(self, shape, pixscale, focus=None, width=None, flux=None, beamsize=None):
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
        flux : float/int or astropy Quantity. sets brightness of the array
            NEED TO DECIDE: what default units make sense here? - probably a surface brightness
            
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
        # to do: write a test that passes pixscale with units other than km
        
        if flux is None:
            flux = self.flux
        if width is None:
            width = self.width   
        pixscale = u.Quantity(pixscale, u.km)

        # put the ring onto a 2-D array using EllipticalAnnulus
        if focus is None:
            focus = (shape[0]/2.0, shape[1]/2.0)
            
        ann = self.as_elliptical_annulus(focus, pixscale, width)
        arr_sharp = ann.to_mask(method='exact').to_image(shape)
        
        if beamsize is None:
            return arr_sharp
        else:
            # make the Gaussian beam. convert FWHM to sigma
            beam = convolution.Gaussian2DKernel(beamsize[0] / 2.35482004503,
                                                beamsize[1] / 2.35482004503, 
                                                Angle(beamsize[2], unit=u.deg))
            return convolution.convolve_fft(arr_sharp, beam)

        
class RingSystemModelObservation:
    
    def __init__(self, 
                planet, 
                location=None, 
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
        location : array-like, or `~astropy.coordinates.EarthLocation`, optional
            Observer's location as a
            3-element array of Earth longitude, latitude, altitude, or
            `~astropy.coordinates.EarthLocation`.  Longitude and
            latitude should be anything that initializes an
            `~astropy.coordinates.Angle` object, and altitude should
            initialize an `~astropy.units.Quantity` object (with units
            of length).  If ``None``, then the geofocus is used.
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
        
        '''
        planet = planet.lower().capitalize()
        self.planetname = planet
        
        # fix inputs
        if epoch is None:
            raise NotImplementedError("not done yet; please provide epoch for now")
        if location is None:
            raise NotImplementedError("not done yet; please provide location for now")

        # send query to Planetary Ring Node, and query static data table
        node = RingNode()
        self.bodytable, self.ringtable = node.ephemeris(
                    planet=planet, epoch=epoch, location=location, cache=False)
        self.systemtable = self.bodytable.meta
        
        ring_data_source = importlib.resources.open_binary('pylanetary.pylanetary.rings.data', f'{planet}_ring_data.hdf5')
        #ring_static_data = table.Table.read(f'data/{planet}_ring_data.hdf5', format = 'hdf5')
        ring_static_data = table.Table.read(ring_data_source, format = 'hdf5')
        planet_ephem = self.bodytable.loc[planet]
        #self.ob_lat, self.ob_lon = planet_ephem['sub_obs_lat'], planet_ephem['sub_obs_lon']
        
        # query Horizons for the north polar angle
        obj = Horizons(id="799", location='-7', epochs={'start':epoch.to_value('iso'), 'stop':(epoch + 1*u.day).to_value('iso'), 'step':'1d'})
        eph = obj.ephemerides()
        self.np_ang = eph['NPole_ang'][0]
        
        
        # TO DO: change the way the static data tables are read in to be more package-y
        # match the static and ephemeris data for rings using a table merge
        ring_static_data.rename_column('Feature', 'ring')
        self.ringtable = table.join(self.ringtable, ring_static_data, keys='ring', join_type='right')
        self.ringtable.add_index('ring')
        
        if ringnames is None:
            ringnames = list(self.ringtable['ring'])
            
        # make self.ringtable and fluxes contain only the rings in ringnames
        self.ringtable = self.ringtable.loc[ringnames]
        if fluxes == 'default':
            fluxes = list(self.ringtable['Optical Depth'])
            fluxes = [1 - np.exp(-val) for val in fluxes] # optical depth to 1 - transmittance
        
        self.rings = {}
        for i in range(len(ringnames)):
            ringname = ringnames[i]
            flux = fluxes[i]
            try:
                ringparams = self.ringtable.loc[ringname]
            except Exception as e:
                raise ValueError(f"Ring name {ringname} not found in the data table of known rings")
            
            # make a Ring object for each one
            omega = self.np_ang*u.deg + self.ringtable.loc['ring','Epsilon']['ascending node'].filled(0.0)# + self.systemtable['sub_obs_lon']
            i = 90*u.deg + self.systemtable['opening_angle']
            w = self.ringtable.loc['ring','Epsilon']['pericenter'].filled(0.0) # filled() just turns from a masked array, which doesn't pass, to an unmasked array. the fill value is not used

            # many of the less-well-observed rings have masked values
            # for many of these quantities, particularly omega, i, w, or even e. these go to
            # zero when made into floats, so it is ok
            #print(omega, i, w)
            thisring = Ring(ringparams['Middle Boundary (km)'] * u.km, 
                        ringparams['Eccentricity'], 
                        omega, 
                        i, 
                        w, 
                        width = ringparams['Width'], 
                        flux = flux)
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
            
            arr_out += self.rings[ringname].as_2d_array(shape, pixscale, focus=focus, beamsize=None)
        
            
        # run convolution with beam outside loop so it is only done once
        if beamsize is None:
            return arr_out
        else:
            # make the Gaussian beam. convert FWHM to sigma
            beam = convolution.Gaussian2DKernel(beamsize[0] / 2.35482004503,
                                                beamsize[1] / 2.35482004503, 
                                                Angle(beamsize[2], unit=u.deg).to(u.radian).value)
            return convolution.convolve(arr_out, beam)
        
if __name__ == "__main__":
    
    # for simple testing
    import matplotlib.pyplot as plt
    a = 51149 #km
    e = 0.007
    i = 45.0
    omega = 80.0
    w = 0.
    imsize = 300 #px
    pixscale = 500 #km/px
    focus = np.array([imsize/2, imsize/2])
    simple_ring = Ring(a, e, omega, i, w, width=5000)
    
    thetas, wedges = simple_ring.as_azimuthal_wedges((imsize, imsize), focus, pixscale, nwedges=60, width=5000*u.km, n=1e3, z=1)
    
    # check if wedges add up to a full ellipse
    wedges_arr = np.asarray(wedges)
    wedges_sum = np.sum(wedges, axis=0)
    plt.imshow(wedges_sum, origin = 'lower')
    plt.show()
    
    #img = simple_ring.as_2d_array((imsize, imsize), pixscale) #shape (pixels), pixscale (km)
    #plt.imshow(img, origin = 'lower')
    #plt.show()
        