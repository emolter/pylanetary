
from astropy import table
from astroquery.solarsystem.pds import RingNode
from astropy.coordinates import Angle
import astropy.units as u

from photutils import aperture
import numpy as np
from PyAstronomy import pyasl
from collections import OrderedDict

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
    
    def __init__(self, a, e, omega, i, w, width = 0.0, flux = 0.0):
        '''
        model of a planetary ring

        Parameters
        ----------
        a : semimajor axis
        e : eccentricity
        Omega : longitude of ascending node
        i : inclination
        w : argument of periapsis
        width : 
        flux : 
        
        Attributes
        ----------
        a : semimajor axis
        e : eccentricity
        omega : longitude of ascending node
        i : inclination
        w : argument of periapsis
        flux : 
        
        
        Examples
        --------
        
        '''
        self.a = a
        self.e = e
        self.omega = Angle(omega, 'deg')
        self.i = Angle(i, 'deg')
        self.w = Angle(w, 'deg')
        self.width = width
        self.flux = flux
    
            
    def as_elliptical_annulus(shape, pixscale, width, center = None):
        '''
        return elliptical annulus surrounding the ring of the given width
        
        
        '''
        if center is None:
            center = (data.shape[0]/2.0, data.shape[1]/2.0)
        ann = aperture.EllipticalAnnulus(center, 
                            a_in=self.a - width/2., 
                            a_out=self.a + width/2., 
                            b_out= abs(self.a + width/2. * np.sin(self.i)), 
                            b_in= None, 
                            theta = Angle(self.w, 'deg'))
                            
        # test whether the angles coming in here are actually correct
        
        return ann
        
    def as_keplers3rd_wedges(width, n):
        '''
        return n partial elliptical annulus wedges with equal orbital time spent in each
        useful for ring as f(azimuth) because should take out foreshortening correction
        but should check this! what did I do for the paper?
        also perfect data experiments would be good
        '''
        
        # do this later, it's complicated to do right
        
        return ann_list
        
    def as_orbit(T=1, tau=0):
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
        
    def as_2d_array(shape, pixscale, width=None, flux=None, beamsize=None):
        '''
        return a 2-d array that looks like a mock observation
        optional smearing over Gaussian beam
        
        Parameters
        ----------
        shape : tuple, required. output image shape
        pixscale : float/int or astropy Quantity, required. pixel scale
            of the output image. If float/int (i.e. no units specified), then
            kilometers is assumed
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
        if flux is None:
            flux = self.flux
        if width is None:
            width = self.width
        
        # can I leverage EllipticalAnnulus again here?
        
        # use whatever functionality Astropy has for convolving with beam
        
        return

        
class RingSystemModelObservation:
    
    def __init__(self, 
                planet, 
                location=None, 
                obs_time=None, 
                ringnames=None,
                fluxes='default'):
        '''
        make a model of a ring system 

        Parameters
        ----------
        planet: str, required. one of Jupiter, Saturn, Uranus, Neptune
        obs_time : `~astropy.time.Time` object, or str in format YYYY-MM-DD hh:mm, optional.
                If str is provided then UTC is assumed.
                If no obs_time is provided, the current time is used.
        location : array-like, or `~astropy.coordinates.EarthLocation`, optional
            Observer's location as a
            3-element array of Earth longitude, latitude, altitude, or
            `~astropy.coordinates.EarthLocation`.  Longitude and
            latitude should be anything that initializes an
            `~astropy.coordinates.Angle` object, and altitude should
            initialize an `~astropy.units.Quantity` object (with units
            of length).  If ``None``, then the geocenter is used.
        ringnames : list, optional. which rings to include in the model
            if no ringnames provided then all rings are assumed.
            Case-sensitive! Typically capitalized, e.g. "Alpha"
                (for now - annoying to make case-insensitive)
        fluxes : 'default' or list, optional. the fluxes of each ring in ringnames,
            in units of XXXXX
        
        Attributes
        ----------
        planetname : str, name of planet
        rings : dict of ringmodel.Ring objects, with ring names as keys
                note ring names are case-sensitive! Typically capitalized, e.g. "Alpha"
        fluxes : 
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
        systemtable, bodytable, ringtable = RingNode.ephemeris(planet, obs_time=obs_time, location=location)
        ring_static_data = table.Table.read(f'data/{planet}_ring_data.hdf5', format = 'hdf5')
        planet_ephem = bodytable.loc[planet]
        ob_lat, ob_lon = planet_ephem['sub_obs_lat'], planet_ephem['sub_obs_lon']
        # TO DO: change the way the data tables are read in to be more package-y
        
        # match the static and ephemeris data for rings using a table merge
        ring_static_data.rename_column('Feature', 'ring')
        ringtable = table.join(ringtable, ring_static_data, keys='ring', join_type='right')
        ringtable.add_index('ring')
        

        if ringnames is None:
            ringnames = list(ringtable['ring'])
        
        self.rings = {}
        self.fluxes = {}  
        for i in range(len(ringnames)):
            ringname = ringnames[i]
            flux = fluxes[i]
            try:
                ringparams = ringtable.loc[ringname]
            except Exception as e:
                raise ValueError(f"Ring name {ringname} not found in the data table of known rings")
            
            # make a Ring object for each one
            # TO DO: MORE MATH HERE
            omega = Angle(ringparams['ascending node'], 'deg') # plus whatever is needed for Uranus orientation wrt observer
            i = Angle(ringparams['Inclination (deg)'], 'deg') + (90*u.deg - systemtable['opening_angle'])
            w = Angle(ringparams['pericenter'], 'deg') # plus whatever is needed for planet orientation wrt observer
            # note it is fine that many of the less-well-observed rings have masked values
            # for many of these quantities, particularly omega, i, w, or even e. these go to
            # zero when made into floats, so 
            thisring = Ring(ringparams['Middle Boundary (km)'] * u.km, 
                        ringparams['Eccentricity'], 
                        omega, 
                        i, 
                        w, 
                        width = ringparams['Width'], 
                        flux = flux)
            self.fluxes[ringname] = flux
            self.rings[ringname] = thisring
            
        #print(ringtable.loc['Epsilon'])
        
        # make self.ringtable contain only the rings in ringnames
        self.ringtable = ringtable.loc[ringnames]
        # TO DO: does the line above actually work?
        
        
        
    def as_2d_array(shape, pixscale, beamsize=None, fluxes='default'):
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
            
            arr_out += self.rings[ringname].as_2d_array(shape, pixscale, width, beamsize=beamsize)
        
        
    
        return 
        
if __name__ == "__main__":
    
    uranus = RingSystemModelObservation('uranus', ringnames = ['Six', 'Five', 'Four', 'Alpha', 'Beta', 'Eta', 'Gamma', 'Delta', 'Epsilon'])           