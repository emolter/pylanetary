# Licensed under a ??? style license - see LICENSE.rst
import numpy as np
import warnings
import astropy.units as u
from astropy.io import fits
from image_registration.chi2_shifts import chi2_shift
from image_registration.fft_tools.shift import shift2d
from scipy import ndimage
from skimage import feature
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.time import Time

from ..utils import *

'''
To implement
------------
* change to rely on Cartopy
* make lat_lon accept either east or west longitudes with a flag?
    * test on Jupiter (GRS), Io (Loki), and others
* function to write ModelEllipsoid and Nav.reproject() outputs to fits
* test support for 2-D Gaussian beams and measured PSFs
* make better docstrings for everything
'''

def lat_lon(x,y,ob_lon,ob_lat,pixscale_km,np_ang,req,rpol):
    '''
    Projection of an ellipsoid onto a 2-D array with latitude and longitude grid
    
    Parameters
    ----------
    x : 
    y : 
    ob_lon :
    ob_lat : 
    pixscale_km : 
    np_ang : 
    req :
    rpol : 
    
    Returns
    -------
    np.array
        planetographic latitudes, shape ???
    np.array
        planetocentric latitudes, shape ???
    np.array
        West longitudes, shape ???
    
    Examples
    --------
    
    '''
    np_ang = -np_ang
    x1 = pixscale_km*(np.cos(np.radians(np_ang))*x - np.sin(np.radians(np_ang))*y)
    y1 = pixscale_km*(np.sin(np.radians(np_ang))*x + np.cos(np.radians(np_ang))*y)
    olrad = np.radians(ob_lat)
    
    #set up quadratic equation for ellipsoid
    r2 = (req/rpol)**2
    a = 1 + r2*(np.tan(olrad))**2 #second order
    b = 2*y1*r2*np.sin(olrad) / (np.cos(olrad)**2) #first order
    c = x1**2 + r2*y1**2 / (np.cos(olrad))**2 - req**2 #constant

    radical = b**2 - 4*a*c
    #will equal nan outside planet since radical < 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") #suppresses error for taking sqrt nan
        x3s1=(-b+np.sqrt(radical))/(2*a)
        x3s2=(-b-np.sqrt(radical))/(2*a)
    z3s1=(y1+x3s1*np.sin(olrad))/np.cos(olrad)
    z3s2=(y1+x3s2*np.sin(olrad))/np.cos(olrad)
    odotr1=x3s1*np.cos(olrad)+z3s1*np.sin(olrad)
    odotr2=x3s2*np.cos(olrad)+z3s2*np.sin(olrad)
    #the two solutions are front and rear intersections with planet
    #only want front intersection
    
    #tricky way of putting all the positive solutions into one array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") #suppresses error for taking < nan
        odotr2[odotr2 < 0] = np.nan
        x3s2[odotr2 < 0] = np.nan
        z3s2[odotr2 < 0] = np.nan
        odotr1[odotr1 < 0] = odotr2[odotr1 < 0]
        x3s1[odotr1 < 0] = x3s2[odotr1 < 0]
        z3s1[odotr1 < 0] = z3s2[odotr1 < 0]
    
    odotr,x3,z3 = odotr1,x3s1,z3s1
    y3 = x1
    r = np.sqrt(x3**2 + y3**2 + z3**2)
    
    #lon_w = np.degrees(np.arctan(y3/x3)) + ob_lon
    lon_w = np.degrees(np.arctan2(x3,y3)-np.pi/2) + ob_lon 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") #suppresses error for taking < nan
        lon_w[lon_w < 0] += 360
        lon_w = lon_w%360
    lat_c = np.degrees(np.arcsin(z3/r))
    lat_g = np.degrees(np.arctan(r2*np.tan(np.radians(lat_c))))

    return lat_g, lat_c, lon_w
    

def surface_normal(lat_g, lon_w, ob_lon):
    '''
    Computes the normal vector to the surface of the planet.
    Take dot product with sub-obs or sub-sun vector to find cosine of emission angle
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    nx = np.cos(np.radians(lat_g))*np.cos(np.radians(lon_w-ob_lon))
    ny = np.cos(np.radians(lat_g))*np.sin(np.radians(lon_w-ob_lon))
    nz = np.sin(np.radians(lat_g))
    return np.asarray([nx,ny,nz])
    

#def sun_normal(lat_g, lon_w, sun_lon, sun_lat):
#    '''
#    Computes the normal vector to the sun.
#    Taking the dot product of output with sub-sun vector
#    gives the cosine of solar incidence angle
#    
#    Parameters
#    ----------
#    
#    Returns
#    -------
#    
#    To do
#    -----
#    * This has not been tested at all
#    '''
#    nx = np.cos(np.radians(lat_g-sun_lat))*np.cos(np.radians(lon_w-sun_lon))
#    ny = np.cos(np.radians(lat_g-sun_lat))*np.sin(np.radians(lon_w-sun_lon))
#    nz = np.sin(np.radians(lat_g-sun_lat))
#    return np.asarray([nx,ny,nz])


def colocate_diagnostic_plot(model, data, mode):
    '''
    assesses goodness-of-fit of navigation solution from colocate
    
    Parameters
    ----------
    model : np.array, required
        shifted model data
    data : np.array, required
        observed data
    mode : str, required
        method of co-location, one of ["canny", "convolution", or "disk"]
    
    Returns
    -------
    matplotlib figure
    '''
    aspect_ratio = data.shape[0] / data.shape[1]
    if aspect_ratio >= 1:
        szx = 12
        szy = szx/aspect_ratio
    else:
        szy = 12
        szx = szy*aspect_ratio
        
    if (mode == 'convolution') or (mode == 'disk'):
        vmax = np.nanmax(model)
    elif mode == 'canny':
        vmax = 1
    else:
        raise ValueError('mode must be one of convolution, disk, or canny')
        
    fig, ax = plt.subplots(1,1, figsize = (szy, szx))
    
    im = ax.imshow(data - model, origin = 'lower', vmax = vmax)
    ax.set_title('Data minus model')
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Pixel Value')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical', label='flux')
    
    return fig, ax
    
    
def emission_angle(ob_lat, surf_n):
    '''
    Computes the cosine of the emission angle of surface wrt observer
    
    Parameters
    ----------
    ob_lat : float or np.array, required. 
        [deg] sub-observer latitude
    surf_n : np.array, required.
        [x,y,z] surface normal vector at each ob_lat
    
    Returns
    -------
    float or np.array
        cosine of emission angle
    '''
    surf_n /= np.linalg.norm(surf_n, axis=0) # normalize to magnitude 1
    ob = np.asarray([np.cos(np.radians(ob_lat)),0,np.sin(np.radians(ob_lat))])
    mu = np.dot(surf_n.T, ob).T
    return mu
    
        
def limb_darkening(mu, a, law='exp', mu0=None):
    '''
    Parameters
    ----------
    mu : float or array-like, required. 
        [-] cosine of emission angle
    a : float or array-like, required. 
        [-] limb-darkening parameter(s)
        if law=="disc", a is ignored 
        if law=="exp", "linear", or "minnaert", a must have length 1
        if law=="quadratic", "square-root", a must have length 2 such that [a0, a1] are 
        the free parameters in the first and second terms, respectively.
    law : str, optional. default "exp".
        what type of limb darkening law to use. options are:
        disc : no limb darkening applied
        linear : ld = 1 - a * (1 - mu)
        exp : ld = mu**a
        minnaert : ld = mu0**a * mu**(a-1)
        quadratic : ld = 1 - a[0]*(1-mu) - a[1]*(1-mu)**2 
        square-root : ld = 1 - a[0]*(1-mu) - a[1]*(1-np.sqrt(mu))
    mu0 : float or array-like, optional. default None. 
        [-] cosine of solar incidence angle.
        has no effect unless law=="minnaert", in which case it is required.
    
    Returns
    -------
    float or np.array
        limb-darkening value at each input mu value

    References
    ----------
    Overview of published limb darkening laws
    https://www.astro.keele.ac.uk/jkt/codes/jktld.html 
    '''
    # Input check 
    if np.any(mu <0.0) or np.any(mu>1.0): 
        raise ValueError('Cosine of emission angle range [0,1]')

    mu = np.array(mu) #necessary so when floats are passed in, the line mu[bad] = 0 doesnt complain about indexing a float
    idnan = np.isnan(mu)
    mu[idnan] = 0.0

    law = law.lower()
    if law == 'disc' or law == 'disk': 
        ld = mu[mu>0] = 1  
    if law == 'exp' or law == 'exponential':
        ld = mu**a
    elif law == 'linear':
        ld = 1 - a * (1 - mu)
    elif law == 'quadratic':
        ld = 1 - a[0]*(1-mu) - a[1]*(1-mu)**2  
    elif law == 'square-root':
        ld = 1 - a[0]*(1-mu) - a[1]*(1-np.sqrt(mu))   
    elif law == 'minnaert':
        if mu0 is None:
            raise ValueError('mu0 must be specified if law == minnaert')
        ld = mu0**a * mu**(a-1)
    else:
        raise ValueError('limb darkening laws accepted: "disc", "linear", "exp", "minnaert", "quadratic", "square-root" ')
    
    ld = np.array(ld)
    ld[idnan] = np.nan

    return ld


class ModelEllipsoid:
    '''
    Projection of an ellipsoid onto a 2-D array with latitude and longitude grid
    '''
    
    def __init__(self, ob_lon, ob_lat, pixscale_km, np_ang, req, rpol, center=(0.0, 0.0), shape=None, sun_lon = None, sun_lat = None):
        '''
        Parameters
        ----------
        ob_lon : float, required. 
            [deg] sub-observer planetographic longitude
        ob_lat : float, required. 
            [deg] sub-observer planetographic latitude
        np_ang : float, required. 
            [deg] north polar angle
            see JPL Horizons ephemeris tool for detailed descriptions 
            of ob_lon, ob_lat, np_ang
        pixscale_km : float, required. 
            [km] pixel scale
        req : float, required. 
            [km] equatorial radius
        rpol : float, required. 
            [km] polar radius
        center : 2-element array-like, optional. default (0,0). 
            pixel location of center of planet
        shape : 2-element tuple, optional. 
            shape of output arrays.
            if None, shape is just larger than diameter / pixscale
        sun_lon : float, optional, default None
            sub-solar longitude. if None, assume same as ob_lon
        sun_lat : float, optional, default None
            sub_solar latitude. if None, assume same as ob_lat
        
        Attributes
        ----------
        req : float
            see parameters
        rpol : float
            see parameters
        ob_lon : float
            see parameters
        ob_lat : float
            see parameters
        sun_lon : float
            see parameters
        sun_lat : float
            see parameters
        pixscale_km : float
            see parameters
        deg_per_px : float
            approximate size of pixel on planet, in degrees, at sub-observer point
        lat_g : np.array
            shape is roughly 2*max(req, rpol)/pixscale. 
            planetographic latitudes. NaN where off planet disk
        lon_w : np.array
            same shape as lat_g. west longitudes. NaN where off planet disk
        mu : np.array
            same shape as lat_g. cosines of the emission angle. NaN where off planet disk
        surf_n : np.array
            shape (3,x,y) CHECK THIS. Normal vector to the surface 
            of the planet at each pixel. NaN where off planet disk
        sun_n : np.array
            shape (3,x,y), solar normal vectors
        mu0 : np.array
            shape (x,y), cosine of solar incidence angle at each pixel
        
        Examples
        --------
        see notebooks/planetnav-tutorial.ipynb
        '''
        
        self.req, self.rpol = req, rpol
        self.pixscale_km = pixscale_km
        self.ob_lon = ob_lon
        self.ob_lat = ob_lat
        self.np_ang = np_ang
        
        if shape is None:
            sz = int(2*np.ceil(np.max([req, rpol]) / pixscale_km) + 1)
            shape = (sz, sz)
        if sun_lon is None:
            self.sun_lon = ob_lon
        else:
            self.sun_lon = sun_lon
        if sun_lat is None:
            self.sun_lat = ob_lat
        else:
            self.sun_lat = sun_lat
        
        xcen, ycen = int(shape[0]/2), int(shape[1]/2) #pixels at center of planet
        yy = np.arange(shape[0]) - xcen
        xx = np.arange(shape[1]) - ycen
        x,y = np.meshgrid(xx,yy) 
        self.lat_g, self.lat_c, self.lon_w = lat_lon(x,y,ob_lon,ob_lat,pixscale_km,np_ang,req,rpol)
        self.surf_n = surface_normal(self.lat_g, self.lon_w, self.ob_lon)
        self.mu = emission_angle(self.ob_lat, self.surf_n)
        
        # sun geometry
        self.sun_n = surface_normal(self.lat_g, self.lon_w, self.sun_lon)
        self.mu0 = emission_angle(self.ob_lat, self.sun_n)
        
        # solve small numerical issue where mu = 1.0 + epsilon
        self.mu[(self.mu > 1.0)*(self.mu < 1.00001)] = 1.0
        self.mu0[(self.mu0 > 1.0)*(self.mu0 < 1.00001)] = 1.0
        
        avg_circumference = 2*np.pi*((self.req + self.rpol)/2.0)
        self.deg_per_px = self.pixscale_km * (1/avg_circumference) * 360
        
        # TO DO: test lon_e vs lon_w for different planets!
        # different systems are default for different planets!
        
        # TO DO: offset model by center() parameter
        if (center[0] != 0) or (center[1] != 0):
            raise NotImplementedError
        
        
    def __str__(self):
        return f'ModelEllipsoid instance; req={self.req}, rpol={self.rpol}'
        
        
    def ldmodel(self, tb, a, beam=None, law='exp', mu0=None, psf_mode='gaussian'):
        '''
        Make a limb-darkened model disk convolved with the beam
        See docstring of limb_darkening() for options
        
        Parameters
        ----------
        tb : float, required. 
            [flux] brightness temperature of disk at mu=1
        a : float or tuple, required. 
            [-] limb darkening parameter(s) 
        beam : float or tuple or np.array
            units pixels. see docstring of utils.convolve_with_beam
        law : str, optional, default "exp"
            limb darkening law
        mu0 : np.array, optional, default None
            cosine of solar incidence angle.
            if None, code will check if self.mu0 is defined, and use that
            has no effect unless law=="minnaert".
            if self.mu0 undefined, law=="minnaert", and mu0=None, then
            code will fail
        psf_mode : str, optional, default "gaussian"
            mode to use for convolve_with_beam(), options "airy", "gaussian"
        '''
        if mu0 is None:
            mu0=getattr(self, "mu0", None)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ldmodel = limb_darkening(np.copy(self.mu), a, law=law, mu0=mu0)
        ldmodel[np.isnan(ldmodel)] = 0.0
        ldmodel = tb*ldmodel
        if beam is None:
            return ldmodel
        else:
            return convolve_with_beam(ldmodel, beam, mode=psf_mode)   
        
        
    def zonalmodel(lats, lons, tbs, a=0.0):
        '''
        
        '''
        raise NotImplementedError()
        # chris code here
        zm = whatever #does not include limb darkening
        # interpolates  onto self.lat_g, self.lon_w
        ldm = ldmodel(1, a)
        
        return ldm * zm    


class ModelBody(ModelEllipsoid):
    
    '''
    Wrapper to ModelEllipsoid that permits passing an ephemeris
    '''
    
    def __init__(self, body, pixscale, shape=None):
        
        '''
        Parameters
        ----------
        body : pylanetary.utils.Body object, required
        pixscale : float or Quantity, required. 
            [arcsec/px] pixel scale of the input image
        shape : tuple, optional. 
            shape of output arrays.
            if None, shape is just larger than diameter / pixscale
        
        Attributes
        ----------
        body : pylanetary.utils.Body object
            see parameters
        name : str
            Name of input body as read from input Body object
        ephem : Astropy QTable. 
            single line of astroquery.horizons ephemeris 
            as read from utils.Body object.
            must have 'PDObsLon', 'PDObsLat', 'delta', and 'NPole_ang' fields.
            If you want to modify the ephemeris, modify body.ephem
        pixscale_arcsec : float
            [arcsec/px] pixel scale of image; see parameters
        pixscale_km : float
            [km/px] pixel scale of image computed from input pixel scale in arcsec
            and distance from object to body according to body.ephem
        parent_attrs : 
            all attributes of ModelEllipsoid
        '''
        
        self.body = body
        self.name = body.name
        self.ephem = body.ephem
        self.req = body.req.value
        self.rpol = body.rpol.value
        self.pixscale_arcsec = pixscale
        self.pixscale_km = self.ephem['delta']*u.au.to(u.km)*np.radians(self.pixscale_arcsec/3600.)
        
        super().__init__(self.ephem['PDObsLon'],
                    self.ephem['PDObsLat'],
                    self.pixscale_km,
                    self.ephem['NPole_ang'],
                    self.req,self.rpol, shape=shape,
                    sun_lon = self.ephem['PDSunLon'], 
                    sun_lat = self.ephem['PDSunLat'])
        
        
    def __str__(self):
        return f'ModelBody instance; req={self.req}, rpol={self.rpol}, pixscale={self.pixscale}'
    

class Nav(ModelBody):
    '''
    functions for comparing a model ellipsoid with observed 2-d imaging data
    '''    
    
    def __init__(self, data, body, pixscale):
        '''
        Build the model body according to the input ephemeris
        
        Parameters
        ----------
        data : np.array, required.
            2-D image data
        body : pylanetary.utils.Body object, required.
        pixscale: float or Quantity, required. 
            [arcsec] pixel scale of the input image
        
        Attributes
        ----------
        data : np.array
            see parameters
        parent_attrs : 
            all attributes of ModelBody and ModelEllipsoid classes
        
        Examples
        --------
        see notebooks/nav-tutorial.ipynb
        
        Notes
        -----
        - Need to test astropy quantity handling, ensure docstring reflects what really happens
        '''
        
        # TO DO: fix these all to accept Astropy quantities
        self.data = data
        super().__init__(body, pixscale, shape=data.shape)
        
          
    def __str__(self):
        return f'Nav instance; req={self.req}, rpol={self.rpol}, pixscale={self.pixscale}'
    
            
    def colocate(self, mode = 'convolution', diagnostic_plot=True, save_plot=None, **kwargs):
        '''
        Co-locate the model planet with the observed planet
        
        Parameters
        ----------
        mode : str, optional. Default 'convolution'.
            Which method should be used to overlay planet model and data. Choices are:
        
            * 'canny': uses the Canny edge detection algorithm.
                kwargs: 
                    :tb: float, required.
                        [same flux unit as data] brightness temperature of disk at mu=1 
                    :a: float, required.
                        [-] exponential limb darkening param
                    :law: str, optional. default 'exp'
                        type of limb darkening model to use
                    :beam: float, tuple, or np.array, optional. default None.
                        units pixels. see utils.convolve_with_beam
                    :psf_mode: str, optional. default "gaussian"
                        what beam shape to use, options "airy", "gaussian"
                    :low_thresh: float, required.
                        see documentation of skimage.feature.canny
                    :high_thresh: float, required.
                        see documentation of skimage.feature.canny
                    :sigma: int, required.
                        see documentation of skimage.feature.canny
                To find edges of planet disk, typical "good" values are:
                    low_thresh : RMS noise in image
                    high_thresh : approximate flux value of background disk (i.e., cloud-free, volcano-free region)
                    sigma : 5
        
            * 'convolution': takes the shift that maximizes the convolution of model and planet
                kwargs:
                    :tb: float, required.
                        [same flux unit as data] brightness temperature of disk at mu=1 
                    :a: float, required.
                        [-] limb darkening parameter
                    :law: str, optional. default 'exp'
                        type of limb darkening model to use
                    :beam: float, tuple, or np.array, optional. default None.
                        units pixels. see utils.convolve_with_beam
                    :psf_mode: str, optional. default "gaussian"
                        what beam shape to use, options "airy", "gaussian"
                    :err: float
                        per-pixel error in input image
        
            * 'disk': same as convolution

        diagnostic_plot : bool, optional. default True
            do you want the diagnostic plots to be shown
        save_plot : str, optional. default None.
            file path to save the diagnostic plot. if None, does not save.
        
        Returns
        -------
        float
            dx in pixels. best-fit difference in position between model and data
            To shift data to center (i.e., colocated with model), apply a shift of -dx, -dy
            To shift model to data, apply a shift of dx, dy
        float
            dy in pixels
        float
            dxerr in pixels uncertainty in the shift based on the cross-correlation 
            from image_registration.chi2_shift. These should be treated with extreme caution
            because they were designed to work on identical images corrupted by random
            noise, and are being used for non-identical images (model and data)
        float
            dyerr in pixels
        
        Examples
        --------
        need at least one example of each mode here
        
        Notes
        -----
        * sometimes dxerr, dyerr give unrealistic or undefined behavior
        '''
        defaultKwargs={'err':None,'beam':None,'law':'exp', 'psf_mode':'gaussian'}
        kwargs = { **defaultKwargs, **kwargs }

        if (mode == 'convolution') or (mode == 'disk'):
            model = self.ldmodel(kwargs['tb'], kwargs['a'], law=kwargs['law'])
            model = convolve_with_beam(model, kwargs['beam'], mode=kwargs['psf_mode'])
            data_to_compare = self.data 
        elif mode == 'canny':
            model_planet = self.ldmodel(kwargs['tb'], kwargs['a'], law=kwargs['law'])
            if kwargs['beam'] is not None:
                model_planet = convolve_with_beam(model_planet, kwargs['beam'], mode=kwargs['psf_mode'])
            
            edges = feature.canny(self.data, sigma=kwargs['sigma'], low_threshold = kwargs['low_thresh'], high_threshold = kwargs['high_thresh'])
            model = feature.canny(model_planet, sigma=kwargs['sigma'], low_threshold = kwargs['low_thresh'], high_threshold = kwargs['high_thresh'])
            data_to_compare = edges
        
        [dx,dy,dxerr,dyerr] = chi2_shift(model, data_to_compare, err=kwargs['err'])
        
        if diagnostic_plot:
            
            model_shifted = shift2d(model, dx, dy)
            fig, ax = colocate_diagnostic_plot(model_shifted, data_to_compare, mode)
            if save_plot is not None:
                fig.savefig(save_plot, dpi=300)
            plt.show()
            plt.close()
            
        return (dx, dy, dxerr, dyerr)
        
    
    def xy_shift_data(self, dx, dy):
        '''
        FFTshift data by a user-defined amount
        for example, to apply the suggested shift from colocate()
        
        Parameters
        ----------
        dx : float, required
            [pixels] shift in x
        dy : float, required
            [pixels] shift in y
        '''
        self.data = shift2d(self.data,dx,dy)
        
    
    def xy_shift_model(self, dx, dy):
        '''
        shift model (i.e., lat_g, lon_w, mu, and mu0) 
        by recalculating the model with a different center
        as defined by user-defined dx, dy shift
        
        Parameters
        ----------
        dx : float, required
            [pixels] shift in x
        dy : float, required
            [pixels] shift in y
        '''
        shape = self.data.shape
        xcen, ycen = int(shape[0]/2), int(shape[1]/2) #pixels at center of planet
        yy = np.arange(shape[0]) - xcen - dy
        xx = np.arange(shape[1]) - ycen - dx
        x,y = np.meshgrid(xx,yy) 
        self.lat_g, self.lat_c, self.lon_w = lat_lon(x,y,self.ob_lon,self.ob_lat,self.pixscale_km,self.np_ang,self.req,self.rpol)
        self.surf_n = surface_normal(self.lat_g, self.lon_w, self.ob_lon)
        self.mu = emission_angle(self.ob_lat, self.surf_n)
        
        # sun geometry
        self.sun_n = surface_normal(self.lat_g, self.lon_w, self.sun_lon)
        self.mu0 = emission_angle(self.ob_lat, self.sun_n)
        
        # solve small numerical issue where mu = 1.0 + epsilon
        self.mu[(self.mu > 1.0)*(self.mu < 1.00001)] = 1.0
        self.mu0[(self.mu0 > 1.0)*(self.mu0 < 1.00001)] = 1.0
        
        
    def write(self, outstem, header={}, flux_unit=''):
        '''
        Writes navigated data to multi-extension fits
        
        Parameters
        ----------
        outstem : str, required.
            stem of filenames to write
        header : dict, optional, default {}
            dictionary of header info to put into hdul[0].header
        flux_unit : str, optional, default ""
            unit of flux to put in output fits header
        
        Writes
        ------
        fits file
        
        Notes
        -----
        hdul[0] contains the header, data is empty
        hdul[1] contains the data
        hdul[2] contains latitudes
        hdul[3] contains longitudes
        hdul[4] contains emission angles
        hdul[5] contains solar incidence angles
        
        References
        ----------
        NAV multi-extension fits file format originally pioneered by Mike Wong
        e.g. https://doi.org/10.3847/1538-4365/ab775f
        '''
        
        hdu0 = fits.PrimaryHDU()
        
        # turn ephem into OPAL-like header
        # allow to be overwritten by input header dict
        header_default = {
            'SIMPLE':True,
            'BITPIX':-32,
            'NAXIS':0,
            'EXTEND':True,
            'NEXTEND':6,
            'TRG_ROT':self.ephem['NPole_ang'],
            'TRG_RA':self.ephem['RA'],
            'TRG_DEC':self.ephem['DEC'],
            'TRG_R_A':self.req, 
            'TRG_R_B':self.rpol,
            'TRG_LON':self.ephem['PDObsLon'],
            'TRG_LAT':self.ephem['PDObsLat'],
            'SUN_LAT':self.ephem['PDSunLat'],
            'TRG_PHAS':self.ephem['alpha'],
            'TRG_R':self.ephem['r'],
            'TRG_D':self.ephem['delta'],
        }
        header_out = {**header_default, **header} #gives priority to input header dict
        hdu0.header = fits.Header(header_out)
        
        # set up metadata for hdul[1-5]
        hdulist = [hdu0]
        data_list = [self.data, 
                    self.lat_g, 
                    self.lon_w, 
                    np.rad2deg(np.arccos(self.mu)), 
                    np.rad2deg(np.arccos(self.mu0)),]
        extnames_list = ['DATA', 'LAT', 'LON', 'EMI', 'INC']
        units_list = [flux_unit, 'DEGREES', 'DEGREES', 'DEGREES', 'DEGREES']
        desc_list = ['', 
                    'planetographic latitude', 
                    'System III longitude', 
                    'emission angle', 
                    'incidence angle']
        date = Time.now()
        date.format = 'iso'
        date = date.iso[:10]
        
        # generate hdul[1-5] headers and data
        for i in range(len(data_list)):
            
            data = data_list[i]
            hdr = fits.Header()
            hdr['XTENSION'] = ('IMAGE', 'Image extension')
            hdr['BITPIX'] = (-32, 'IEEE single precision floating point')
            hdr['NAXIS'] = 2
            hdr['NAXIS1'] = data.shape[0]
            hdr['NAXIS2'] = data.shape[1]
            hdr['DATATYPE'] = ('REAL*4', 'Type of data')
            hdr['DATE'] = (date, 'date the navigation solution was written')
            hdr['INHERIT'] = ('T', 'inherit the primary header')
            hdr['BUNIT'] = (units_list[i], desc_list[i])
            hdu = fits.ImageHDU(data=data, header=hdr, name=extnames_list[i])
            hdulist.append(hdu)
        
        # write it
        hdul = fits.HDUList(hdulist)
        hdul.writeto(outstem, overwrite=True)
        
    
    def reproject(self, pixscale_arcsec = None, interp = 'cubic'):
        '''
        Projects the data onto a flat x-y grid according to self.lat_g, self.lon_w
        This function only works properly if self.lat_g and self.lon_w 
        are centered with respect to self.data; for instance, 
        if ONE of xy_shift_data or xy_shift_model has been applied 
        using the dx, dy output of colocate()
        
        Parameters
        ----------
        pixscale_arcsec : float, optional. default self.pixscale_arcsec
            [arcsec] Pixel scale of output
            If not set, output data will have the same pixel scale as the input data 
            at the sub-observer point. 
            Note that everywhere else will be super-sampled.
        interp : str, optional. default 'cubic'
            type of interpolation to do between pixels in the projection.
        
        Returns
        -------
        np.array
            the projected data 
        np.array
            cosine of the emission angle (mu) 
            at each pixel in the projection
        '''
        
        #determine the number of pixels in resampled image
        if pixscale_arcsec is None:
            pixscale_arcsec = self.pixscale_arcsec
        npix_per_degree = (1/self.deg_per_px) * (self.pixscale_arcsec / pixscale_arcsec) # (old pixel / degree lat) * (new_pixscale / old_pixscale) = new pixel / degree lat
        npix = int(npix_per_degree * 180) + 1 #(new pixel / degree lat) * (degree lat / planet) = new pixel / planet
        print('New image will be %d by %d pixels'%(2*npix + 1, npix))
        print('Pixel scale %f km = %f pixels per degree'%(self.pixscale_km * (pixscale_arcsec / self.pixscale_arcsec), npix_per_degree))
        
        #create new lon-lat grid
        extra_wrap_dist = 180
        newlon, newlat = np.arange(-extra_wrap_dist,360 + extra_wrap_dist, 1/npix_per_degree), np.arange(-90,90, 1/npix_per_degree)
        gridlon, gridlat = np.meshgrid(newlon, newlat)
        nans = np.isnan(self.lon_w.flatten())
        def input_helper(arr, nans):
            '''removing large region of NaNs speeds things up significantly'''
            return arr.flatten()[np.logical_not(nans)]
        inlon, inlat, indat = input_helper(self.lon_w, nans), input_helper(self.lat_g, nans), input_helper(self.data, nans)

        #fix wrapping by adding dummy copies of small lons at > 360 lon
        inlon_near0 = inlon[inlon < extra_wrap_dist]
        inlon_near0 += 360
        inlon_near360 = inlon[inlon > 360 - extra_wrap_dist]
        inlon_near360 -= 360
        inlon_n = np.concatenate((inlon_near360, inlon, inlon_near0))
        inlat_n = np.concatenate((inlat[inlon > 360 - extra_wrap_dist], inlat, inlat[inlon < extra_wrap_dist]))
        indat_n = np.concatenate((indat[inlon > 360 - extra_wrap_dist], indat, indat[inlon < extra_wrap_dist]))

        #do the regridding
        datsort = griddata((inlon_n, inlat_n), indat_n, (gridlon, gridlat), method = interp)
        
        #trim extra data we got from wrapping
        wrap_i_l = len(gridlon[0][gridlon[0] < 0]) - 1
        wrap_i_u = len(gridlon[0][gridlon[0] >= 360])
        datsort = datsort[:,wrap_i_l:-wrap_i_u]
        gridlon = gridlon[:,wrap_i_l:-wrap_i_u]
        gridlat = gridlat[:,wrap_i_l:-wrap_i_u]
        
        # make far side of planet into NaNs
        snorm = surface_normal(gridlat, gridlon, self.ob_lon)
        #emang = emission_angle(self.ob_lat, snorm).T
        emang = emission_angle(self.ob_lat, snorm)
        farside = np.where(emang < 0.0)
        datsort[farside] = np.nan
        emang[farside] = np.nan
        projected = datsort
        mu_projected = emang
        
        self.projected_data = projected
        self.projected_mu = mu_projected

        return projected, mu_projected

