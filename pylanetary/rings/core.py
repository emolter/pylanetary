
from astropy import table
from astroquery.solarsystem.pds import RingNode
from astroquery.solarsystem.jpl import Horizons
from astropy.coordinates import Angle
import astropy.units as u
from astropy import convolution

from photutils import aperture
import numpy as np
from PyAstronomy import pyasl
from collections import OrderedDict
from scipy.spatial.transform import Rotation

'''
goal: 
    pull together JPL Horizons query, ring node query, 
    and static ring data from data/ to make model ring
    systems as observed from any location
    
    is there a better/more astropy-like library to import for keplerian ellipses than PyAstronomy?
        yes, but it is annoying to use
        much later: get an undergrad to make Keplerian ellipse module of Astropy
'''


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
        return f'Ring instance; a={self.a}, e={self.e}, i={self.i}, width={self.width}'
    
            
    def as_elliptical_annulus(self, focus, pixscale, width=None):
        '''
        return elliptical annulus surrounding the ring of the given width
        in pixel space
        
        focus : tuple, required. location of planet (one ellipse focus) in pixels
        pixscale : float or Quantity, required. assumes km if not an astropy Quantity
        width : 
        '''
        
        if width is None:
            width = self.width
        pixscale = u.Quantity(pixscale, unit=u.km)
            
        # convert between focus and center of ellipse based on eccentricity
        c = self.e*self.a / pixscale # distance of focus from center
        cy = c*np.sin(self.w.radian)
        cx = c*np.cos(self.w.radian)
        center = (focus[0] - cx, focus[1] - cy)
        
        def b_from_ae(a,e):
            return a*np.sqrt(1 - e**2)
        
        a_in = ((self.a - width/2.)/pixscale).value
        a_out = ((self.a + width/2.)/pixscale).value
        b_out = b_from_ae(a_out,self.e) # account for eccentricity
        b_out = abs(b_out * np.cos(self.i)).value # account for inclination, which effectively shortens b even more
        ann = aperture.EllipticalAnnulus(center, 
                            a_in=a_in, 
                            a_out=a_out, 
                            b_out=b_out, 
                            b_in= None, 
                            theta = self.w.to(u.radian).value)
                            
        # test whether the angles coming in here are actually correct
        
        return ann
        
    def as_keplers3rd_wedges(self, width, n):
        '''
        return n partial elliptical annulus wedges with equal orbital time spent in each
        useful for ring as f(azimuth) because should take out foreshortening correction
        but should check this! what did I do for the paper?
        also perfect data experiments would be good
        '''
        
        # do this later, it's complicated to do right
        
        return ann_list
        
    def as_orbit(self, T=1, tau=0):
        '''
        make a PyAstronomy.KeplerEllipse object at the ring's orbit
        to get position of ring particles as a function of time
        
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
        
        ke = pyasl.KeplerEllipse(self.a, T, tau = self.tau, e = self.e, Omega = self.omega, i = self.i, w = self.w)
        return ke
        
    def as_2d_array(self, shape, pixscale, opening_angle=90.*u.deg, focus=None, width=None, flux=None, beamsize=None):
        '''
        return a 2-d array that looks like a mock observation
        optional smearing over Gaussian beam
        
        Parameters
        ----------
        shape : tuple, required. output image shape
        pixscale : float/int or astropy Quantity, required. pixel scale
            of the output image. If float/int (i.e. no units specified), then
            kilometers is assumed
        opening_angle : astropy Angle 
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
        if fluxes == 'default':
            fluxes = list(self.ringtable['Optical Depth'])
        
        Attributes
        ----------
        planetname : str, name of planet
        rings : dict of ringmodel.Ring objects, with ring names as keys
                note ring names are case-sensitive! Typically capitalized, e.g. "Alpha"
        ringtable : table of ephemeris data as well as time-invariant parameters for 
        
        
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
        
        # query planetary ring node and static data
        self.systemtable, self.bodytable, ringtable = RingNode.ephemeris(planet, epoch=epoch, location=location)
        ring_static_data = table.Table.read(f'data/{planet}_ring_data.hdf5', format = 'hdf5')
        planet_ephem = self.bodytable.loc[planet]
        #self.ob_lat, self.ob_lon = planet_ephem['sub_obs_lat'], planet_ephem['sub_obs_lon']
        
        # TO DO: change the way the static data tables are read in to be more package-y
        
        # match the static and ephemeris data for rings using a table merge
        ring_static_data.rename_column('Feature', 'ring')
        ringtable = table.join(ringtable, ring_static_data, keys='ring', join_type='right')
        ringtable.add_index('ring')
        
        if ringnames is None:
            ringnames = list(ringtable['ring'])
            
        # make self.ringtable and fluxes contain only the rings in ringnames
        self.ringtable = ringtable.loc[ringnames]
        if fluxes == 'default':
            fluxes = list(self.ringtable['Optical Depth'])
        
        self.rings = {}
        for i in range(len(ringnames)):
            ringname = ringnames[i]
            flux = fluxes[i]
            try:
                ringparams = ringtable.loc[ringname]
            except Exception as e:
                raise ValueError(f"Ring name {ringname} not found in the data table of known rings")
            
            # make a Ring object for each one
            # TO DO: MORE MATH HERE
            omega = ringparams['ascending node'] # CHECK THIS
            i = u.Quantity(ringparams['Inclination (deg)'], unit=u.deg) # CHECK THIS
            w = ringparams['pericenter'] # CHECK THIS

            # many of the less-well-observed rings have masked values
            # for many of these quantities, particularly omega, i, w, or even e. these go to
            # zero when made into floats, so it is ok
            thisring = Ring(ringparams['Middle Boundary (km)'] * u.km, 
                        ringparams['Eccentricity'], 
                        omega, 
                        i, 
                        w, 
                        width = ringparams['Width'], 
                        flux = flux)
            self.rings[ringname] = thisring
            
        #print(ringtable.loc['Epsilon'])
        
        
        # TO DO: does the line above actually work?
        
        
        
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
        
        ## project this onto the observer plane using ring opening angle and north pole angle
        r = Rotation('xyz', [self.systemtable['sub_obs_lon'], 0, 90*u.deg - self.systemtable[opening_angle]], degrees=True)
        rvec = r.as_rotvec()
        # TO DO: finish this!
        
            
        # run convolution with beam outside loop so it is only done once
        if beamsize is None:
            return arr_out
        else:
            # make the Gaussian beam. convert FWHM to sigma
            beam = convolution.Gaussian2DKernel(beamsize[0] / 2.35482004503,
                                                beamsize[1] / 2.35482004503, 
                                                Angle(beamsize[2], unit=u.deg))
            return convolution.convolve(arr_out, beam)
        
if __name__ == "__main__":
    
    uranus_rings = RingSystemModelObservation('uranus',
                     epoch='2022-05-03 11:50',
                     ringnames = ['Six', 'Five', 'Four', 'Alpha', 'Beta', 'Eta', 'Gamma', 'Delta', 'Epsilon'])
    obs = uranus_rings.as_2d_array((500, 500), 300*u.km, beamsize = (7,4,30*u.degree)) 
    
    import matplotlib.pyplot as plt
    plt.imshow(obs, origin = 'lower')
    plt.show()         