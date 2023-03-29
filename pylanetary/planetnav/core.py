# Licensed under a ??? style license - see LICENSE.rst
import numpy as np
import warnings
import astropy.units as u
from image_registration.chi2_shifts import chi2_shift
from image_registration.fft_tools.shift import shiftnd, shift2d
from scipy import ndimage
from skimage import feature
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from ..utils import *

'''
To implement
------------
* make lat_lon accept either east or west longitudes with a flag
    * test on Jupiter (GRS), Io (Loki), and others
* function to write ModelEllipsoid and PlanetNav.reproject() outputs to fits
* test support for 2-D Gaussian beams and measured PSFs
* implement quadratic limb darkening
* why are there two functions for surface normal and sun normal? should surface normal also account for latitude?
'''

def lat_lon(x,y,ob_lon,ob_lat,pixscale_km,np_ang,req,rpol):
    '''
    Projection of an ellipsoid onto a 2-D array with latitude and longitude grid
    
    Parameters
    ----------
    x, y : 
    ob_lon, ob_lat : 
    pixscale_km : 
    np_ang : 
    req, rpol : 
    
    Returns
    -------
    lat_g : planetographic latitudes, shape ???
    lat_c : planetocentric latitudes, shape ???
    lon_w : West longitudes, shape ???
    
    Examples
    --------
    
    To-do list
    ----------
    write lots of tests to ensure all the geometry is correct
    but what to test against?
    this has been working reasonably well for years with real data, but there might
        still be small bugs.
    for example, should figure out *actually* whether longitudes are E or W when
        passed sub-observer longitudes in each formalism.
        should have the option to choose East or West longitudes
    probably need to write some math in this docstring
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
    

def sun_normal(lat_g, lon_w, sun_lon, sun_lat):
    '''Computes the normal vector to the surface of the planet.
    Taking the dot product of output with sub-obs or sub-sun vector
         gives the cosine of emission angle
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    nx = np.cos(np.radians(lat_g-sun_lat))*np.cos(np.radians(lon_w-sun_lon))
    ny = np.cos(np.radians(lat_g-sun_lat))*np.sin(np.radians(lon_w-sun_lon))
    nz = np.sin(np.radians(lat_g-sun_lat))
    return np.asarray([nx,ny,nz])
    
    
def emission_angle(ob_lat, surf_n):
    '''
    Computes the cosine of the emission angle of surface wrt observer
    
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    ob = np.asarray([np.cos(np.radians(ob_lat)),0,np.sin(np.radians(ob_lat))])
    return np.dot(surf_n.T, ob).T
    
    
def limb_darkening(mu, a, law='exp', mu0=None):
    '''
    Parameters
    ----------
    mu: float or array-like, required. cosine of emission angle
    a: float or array-like, required. limb-darkening parameter(s)
        if law=="exp", "linear", or "minnaert", a must have length 1
        if law=="quadratic", a must have length 2 such that [a0, a1] are 
            the free parameters in the ??? and ??? terms, respectively.
    law: str, optional. default "exp".
        what type of limb darkening law to use. options are:
        linear:
        quadratic:
        exp:
        minnaert:
    mu0: float or array-like, optional. default None. cosine of solar incidence angle.
        has no effect unless law=="minnaert", in which case it is required.
    
    Returns
    -------
    ld: limb-darkening value at each input mu value
    
    To-do list
    ----------
    add quadratic fits, e.g. Equation 12 of https://doi.org/10.1029/2020EA001254
    support for nonzero solar incidence angles - see JWST Io model
    what should be done when mu > 1 or mu < 0 is passed?
    '''
    
    mu = np.array(mu) #necessary so when floats are passed in, the line mu[bad] = 0 doesnt complain about indexing a float
    bad = np.isnan(mu)
    mu[bad] = 0.0
    if law.lower() == 'exp' or law.lower() == 'exponential':
        ld = mu**a
    elif law.lower() == 'linear':
        ld = 1 - a * (1 - mu)
    elif law.lower() == 'quadratic':
        raise NotImplementedError()
    elif law.lower() == 'minnaert':
        if mu0 is None:
            raise ValueError('mu0 must be specified if law == minnaert')
        ld = mu0**a * mu**(a-1)
    else:
        raise ValueError('limb darkening laws accepted: "linear", "exp"')
    ld = np.array(ld)
    ld[bad] = np.nan
    return ld


class ModelEllipsoid:
    '''
    Projection of an ellipsoid onto a 2-D array with latitude and longitude grid
    '''
    
    def __init__(self, ob_lon, ob_lat, pixscale_km, np_ang, req, rpol, center=(0.0, 0.0), shape=None):
        '''
        Parameters
        ----------
        ob_lon, ob_lat, np_ang : float, required. 
            sub-observer longitude, latitude, np_ang in degrees
            see JPL Horizons ephemeris tool for detailed descriptions
        pixscale_km : float, required. pixel scale in km
        req, rpol : float, required. planet equatorial, polar radius in km
        center : 2-element array-like, optional. default (0,0). 
            pixel location of center of planet
        shape: 2-element tuple, optional. shape of output arrays.
            if None, shape is just larger than diameter / pixscale
        
        Attributes
        ----------
        req, rpol : see parameters
        ob_lon, ob_lat : see parameters
        pixscale_km : see parameters
        deg_per_px : approximate size of pixel on planet, in degrees, at sub-observer point
        lat_g : 2-D array, shape is roughly 2*max(req, rpol)/pixscale. 
            planetographic latitudes. NaN where off planet disk
        lon_w : 2-D array, same shape as lat_g. west longitudes. NaN where off planet disk
        mu : 2-D array, same shape as lat_g. cosines of the emission angle. NaN where off planet disk
        surf_n : 3-D array, shape (3,x,y) CHECK THIS. Normal vector to the surface 
            of the planet at each pixel. NaN where off planet disk
        
        Examples
        --------
        see notebooks/planetnav-tutorial.ipynb
        '''
        
        # TO DO: handle Astropy units here
        self.req, self.rpol = req, rpol
        self.pixscale_km = pixscale_km
        self.ob_lon = ob_lon
        self.ob_lat = ob_lat
        self.np_ang = np_ang
        
        if shape is None:
            sz = int(2*np.ceil(np.max([req, rpol]) / pixscale_km) + 1)
            shape = (sz, sz)
        
        xcen, ycen = int(shape[0]/2), int(shape[1]/2) #pixels at center of planet
        xx = np.arange(shape[0]) - xcen
        yy = np.arange(shape[1]) - ycen
        x,y = np.meshgrid(xx,yy)
        self.lat_g, self.lat_c, self.lon_w = lat_lon(x,y,ob_lon,ob_lat,pixscale_km,np_ang,req,rpol)
        self.surf_n = surface_normal(self.lat_g, self.lon_w, self.ob_lon)
        self.mu = emission_angle(self.ob_lat, self.surf_n)
        
        # TO DO: test lon_e vs lon_w for different planets!
        # different systems are default for different planets!
        
        # TO DO: offset model by center() parameter
        if (center[0] != 0) or (center[1] != 0):
            raise NotImplementedError
        
        
    def __str__(self):
        return f'ModelEllipsoid instance; req={self.req}, rpol={self.rpol}'
        
        
    def write(self, outstem):
        '''
        writes latitudes, longitudes, 
        
        Parameters
        ----------
        outstem : stem of filenames to write
        '''
        raise NotImplementedError
        ### need to rewrite this to make fits files from scratch
        ### with some useful header information
        hdulist_out = self.im.hdulist
        #latitudes
        hdulist_out[0].header['OBJECT'] = target+'_LATITUDES'
        hdulist_out[0].data = self.lat_g
        hdulist_out[0].writeto(lead_string + '_latg.fits', overwrite=True)
        #longitudes
        hdulist_out[0].header['OBJECT'] = target+'_LONGITUDES'
        hdulist_out[0].data = self.lon_w
        hdulist_out[0].writeto(lead_string + '_lonw.fits', overwrite=True)
        #longitudes
        hdulist_out[0].header['OBJECT'] = target+'_MU'
        hdulist_out[0].data = self.mu
        hdulist_out[0].writeto(lead_string + '_mu.fits', overwrite=True)
        
        return


class ModelBody(ModelEllipsoid):
    
    '''
    docstring
    '''
    
    def __init__(self, ephem, req, rpol, pixscale, shape=None):
        
        self.pixscale_arcsec = pixscale
        self.ephem = ephem
        self.pixscale_km = self.ephem['delta']*u.au.to(u.km)*np.radians(self.pixscale_arcsec/3600.)
        
        super().__init__(self.ephem['PDObsLon'],
                    self.ephem['PDObsLat'],
                    self.pixscale_km,
                    self.ephem['NPole_ang'],
                    req,rpol, shape=shape)
        
        avg_circumference = 2*np.pi*((self.req + self.rpol)/2.0)
        self.deg_per_px = self.pixscale_km * (1/avg_circumference) * 360
        
        
    def __str__(self):
        return f'ModelBody instance; req={self.req}, rpol={self.rpol}, pixscale={self.pixscale}'
    
        
    def ldmodel(self, tb, a, beam = None, law='exp'):
        '''
        Make a limb-darkened model disk convolved with the beam
        
        Parameters
        ----------
        tb : float. brightness temperature of disk at mu=1
        a : float, required. limb darkening parameter
        beam : float/int, 3-element array-like, or 2-D psf array, optional.
            FWHM of Gaussian beam with which to convolve the observation
            units of fwhm are number of pixels.
            - if 3-element array-like, has form (FWHM_X, FWHM_Y, POSITION_ANGLE)
            units of position angle are assumed degrees unless astropy Angle is passed
            - if float/int, this is FWHM of assumed circular beam
            - if 2-D array, assumes beam is set to the full PSF
            - if no beamsize is specified, will make infinite-resolution
        law : str, optional. options 'exp' or 'linear'. type of limb darkening law to use
        '''
        ## TO DO: make this allow a 2-D Gaussian beam!
        
        ldmodel = limb_darkening(np.copy(self.mu), a, law=law)
        ldmodel[np.isnan(ldmodel)] = 0.0
        ldmodel = tb*ldmodel
        if beam is None:
            return ldmodel
        elif np.array(beam).size == 1: #FWHM
            fwhm = beam / self.pixscale_arcsec
            return convolve_with_beam(ldmodel, fwhm)
            
        elif np.array(beam).size == 3: #bmaj, bmin, theta
            beam = (beam[0]/self.pixscale_arcsec, beam[1]/self.pixscale_arcsec, beam[2])
            return convolve_with_beam(ldmodel, beam)
            
        elif len(np.array(beam).shape) == 2: #full PSF
            return convolve_with_beam(ldmodel, beam)
        else:
            raise ValueError("beam must be a positive float, 3-element array-like (fwhm_x, fwhm_y, theta_deg), or 2-D array representing the PSF.")    
        
        
    def zonalmodel():
        raise NotImplementedError()
        return  
        

class Nav(ModelBody):
    '''
    use model planet ellipsoid to navigate image data for a planetary body
    
    '''    
    
    def __init__(self, data, ephem, req, rpol, pixscale):
        '''
        Build the planet model according to the observation parameters
        Model (i.e. self.lat_g, self.lon_w, self.mu) are same shape as data and
             initially at center of array. 
        
        Parameters
        ----------
        data : 2-D np array, required. 
        ephem : Astropy QTable, required. one line from an astroquery.horizons ephemeris. needs to at least have
            the 'PDObsLon', 'PDObsLat', 'delta', and 'NPole_ang' fields
            it's ok to pass multiple lines, but will extract relevant fields from the 0th line
        req : float or Quantity object, required. equatorial radius of planet
        rpol : float or Quantity object, required. polar radius of planet
        pixscale : float or Quantity object, required. pixel scale of the images.
            if float, assumes units of arcsec
        
        Attributes
        ----------
        data : see parameters
        req, rpol : see parameters
        pixscale_arcsec : pixel scale of image in arcsec
        pixscale_km : pixel scale of image in km based on data in ephem
        ephem : see parameters
        deg_per_px : approximate size of pixel on planet, in degrees, at sub-observer point
        lat_g : 2-D array, same shape as data. planetographic latitudes. NaN where off planet disk
        lon_w : 2-D array, same shape as data. west longitudes. NaN where off planet disk
        mu : 2-D array, same shape as data. cosines of the emission angle. NaN where off planet disk
        surf_n : 3-D array, shape (3,x,y) CHECK THIS. Normal vector to the surface 
            of the planet at each pixel. NaN where off planet disk
        
        Examples
        --------
        see notebooks/nav-tutorial.ipynb
        '''
        
        # TO DO: fix these all to accept Astropy quantities
        self.data = data
        super().__init__(ephem, req, rpol, pixscale, shape=data.shape)
        
          
    def __str__(self):
        return f'PlanetNav instance; req={self.req}, rpol={self.rpol}, pixscale={self.pixscale}'
    
            
    def colocate(self, mode = 'convolution', diagnostic_plot=True, save_plot=None, **kwargs):
        '''
        Co-locate the model planet with the observed planet
            using any of several different methods
        
        Parameters
        ----------
        mode : str, optional. Which method should be used to overlay
            planet model and data. Choices are:
            'canny': uses the Canny edge detection algorithm...
                kwargs: {'low_thresh':float, ;high_thresh':float, 'sigma':float}
                    see documentation of skimage.feature.canny for explanation
                    To find edges of planet disk, typical "good" values are:
                    low_thresh : RMS noise in image
                    high_thresh : approximate flux value of background disk (i.e., cloud-free, volcano-free region)
                    sigma : 5
            'manual': shift by a user-defined number of pixels in x and y
                kwargs: shift_x, shift_y
            'convolution': takes the shift that maximizes the convolution of model and planet
                kwargs: {'tb':float, 'a':float, 'beamsize':float}
                    tb : brightness temperature of disk at mu=1, 
                    a : limb darkening param, beam in arcsec
                    beam : see ldmodel docstring
                    err : per-pixel error in input image
            'disk': same as convolution
            default 'convolution'
        diagnostic_plot: bool, optional. default True
            do you want the diagnostic plots to be shown
        save_plot: str or None, optional. 
            if str, file path to save the diagnostic plot. if None, does not save.
            does nothing if diagnostic_plot == False
        
        Returns
        -------
            dx, dy: best-fit difference in position between model and data, in pixel units
                To shift data to center (i.e., colocated with model), apply a shift of -dx, -dy
                To shift model to data, apply a shift of dx, dy
            dxerr, dyerr: uncertainty in the shift based on the cross-correlation 
                from image_registration.chi2_shift
        
        Examples
        --------
        need at least one example of each mode here
        
        To Do
        -----
        * improve the Canny edge-detection according to the skimage tutorial
        * make dxerr, dyerr realistic, or remove this option
            * play with adding per-pixel error to chi2_shift call
        '''
        defaultKwargs={'err':None,'beam':None}
        kwargs = { **defaultKwargs, **kwargs }

        if (mode == 'convolution') or (mode == 'disk'):
            model = self.ldmodel(kwargs['tb'], kwargs['a'], beam=kwargs['beam'], law='exp')
            data_to_compare = self.data 
        elif mode == 'canny':
            #model_planet = ~np.isnan(self.mu) #flat disk model
            model_planet = self.ldmodel(kwargs['tb'], kwargs['a'], beam=kwargs['beam'], law='exp')
            
            edges = feature.canny(self.data, sigma=kwargs['sigma'], low_threshold = kwargs['low_thresh'], high_threshold = kwargs['high_thresh'])
            model = feature.canny(model_planet, sigma=kwargs['sigma'], low_threshold = kwargs['low_thresh'], high_threshold = kwargs['high_thresh'])
            data_to_compare = edges
        [dx,dy,dxerr,dyerr] = chi2_shift(model, data_to_compare, err=kwargs['err'])
        
        if diagnostic_plot:
            
            model_shifted = shift2d(model, dx, dy)
            
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 5))
            
            ax0.imshow(data_to_compare, origin = 'lower')
            if mode == 'canny':
                ax0.set_title('data edges')
            elif mode == 'lddisk' or mode == 'disk':
                ax0.set_title('data')
            
            ax1.imshow(model, origin = 'lower')
            if mode == 'canny':
                ax1.set_title(r'Canny filter, $\sigma=$%d'%kwargs['sigma'])
            elif mode == 'lddisk' or mode == 'disk':
                ax1.set_title(r'Limb-darkened disk model')
            
            #ax2.imshow(model_shifted, origin = 'lower', alpha = 0.5)
            ax2.imshow(data_to_compare - model_shifted, origin = 'lower', alpha = 0.5)
            ax2.set_title('Data minus model')
            
            if save_plot is not None:
                plt.savefig(save_plot)
            plt.show()
            plt.close()
            
        return (dx, dy, dxerr, dyerr)
        
    
    def xy_shift_data(self, dx, dy):
        '''
        simple function to FFTshift data by a user-defined amount
        for example, to apply the suggested shift from colocate()
        
        Parameters
        ----------
        dx, dy : floats, required. x,y shift in units of pixels
        '''
        self.data = shift2d(self.data,dx,dy)


    def xy_shift_model(self, dx, dy):
        '''
        simple function to FFTshift model (i.e., lat_g, lon_w, and mu) 
            by a user-defined amount
        for example, to apply the suggested shift from colocate()
        
        Parameters
        ----------
        dx, dy : floats, required. x,y shift in units of pixels
        '''
        
        good = ~np.isnan(self.mu)
        good_shifted = shift2d(good,dx,dy)
        bad_shifted = good_shifted < 0.1
        outputs = []
        for arr in [self.mu, self.lon_w, self.lat_g]:
            
            arr[~good] = 0.0
            arr_shifted = shift2d(arr,dx,dy)
            arr_shifted[bad_shifted] = np.nan
            outputs.append(arr_shifted)
            
        self.mu, self.lon_w, self.lat_g = outputs 
        
    
    def reproject(self, pixscale_arcsec = None, interp = 'cubic'):
        '''
        Projects the data onto a flat x-y grid according to self.lat_g, self.lon_w
        This function only works properly if self.lat_g and self.lon_w 
            are centered with respect to self.data; for instance, 
            if ONE of xy_shift_data or xy_shift_model has been applied 
            using the dx, dy output of colocate()
        
        Parameters
        ----------
        outstem: str, optional. stem of output filenames. if not set, will not save
        pixscale_arcsec: float, optional. Pixel scale of output in arcseconds. 
            If not set, output data will have the same pixel scale as the input data 
            at the sub-observer point. 
            Note that everywhere else will be super-sampled.
        interp: str, optional. type of interpolation to do between pixels in the projection.
            default "cubic"
        
        Returns
        -------
        projected: 2-D numpy array. the projected data 
        mu_projected: 2-D numpy array. the cosine of the emission angle (mu) 
            at each pixel in the projection
        
        Outputs
        -------
        outstem+"_proj.fits" : fits file containing projected data
        outstem+:_mu_proj.fits" : fits file containing mu values of projected data
        
        To-do list
        ----------
        * reproject to any geometry, or at least to polar geometry
        * why does this look awful when we shift model instead of data?
        
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

