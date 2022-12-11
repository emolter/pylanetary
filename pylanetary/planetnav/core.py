# Licensed under a ??? style license - see LICENSE.rst
import numpy as np
import warnings
import astropy.units as u
from image_registration.chi2_shifts import chi2_shift
from image_registration.fft_tools.shift import shiftnd, shift2d
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# fix imports later
from pylanetary.pylanetary.utils.core import *

'''
To implement
------------
* make lat_lon accept either east or west longitudes with a flag
* test PlanetNav.reproject()
* writing ModelPlanetEllipsoid and PlanetNav.reproject() outputs to fits
* support for 2-D Gaussian beams and measured PSFs in ldmodel() and elsewhere
'''

def lat_lon(x,y,ob_lon,ob_lat,pixscale_km,np_ang,req,rpol):
    '''
    Projection of an ellipsoid onto a 2-D array with latitude and longitude grid
    
    Parameters
    ----------
    
    Returns
    -------
    
    Examples
    --------
    
    To-do list
    ----------
    write lots of tests to ensure all the geometry is correct
    but what to test against?
    this has been working well for years against real data, but there might
        still be small bugs.
    for example, should figure out *actually* whether longitudes are E or W when
        passed sub-observer longitudes in each formalism.
        should have the option to choose East or West longitudes
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
    
    
def limb_darkening(mu, a, law='exp'):
    '''
    Parameters
    ----------
    mu: float or array-like, required. cosine of emission angle
    a: float or array-like, required. limb-darkening parameter(s)
        if law=="exp" or law=="linear", a must have length 1
        if law=="quadratic", a must have length 2 such that [a0, a1] are 
            the free parameters in the ??? and ??? terms, respectively.
    
    Returns
    -------
    ld: limb-darkening value at each input mu value
    
    To-do list
    ----------
    add quadratic fits, e.g. Equation 12 of https://doi.org/10.1029/2020EA001254
    '''
    
    bad = np.isnan(mu)
    mu[bad] = 0.0
    if law == 'exp' or law == 'exponential':
        ld = mu**a
        ld[bad] = np.nan
        return ld
    elif law == 'linear':
        ld = 1 - a * (1 - mu)
        ld[bad] = np.nan
        return ld
    elif law == 'quadratic':
        raise NotImplementedError
    else:
        raise ValueError('limb darkening laws accepted: "linear", "exp"')
    

class ModelPlanetEllipsoid:
    '''
    Projection of an ellipsoid onto a 2-D array with latitude and longitude grid
    '''
    
    def __init__(self, shape, ob_lon, ob_lat, pixscale_km, np_ang, req, rpol, center=(0.0, 0.0)):
        '''
        Parameters
        ----------
        shape : 2-element array-like of ints, required. shape of lat_g, lon_w
        ob_lon, ob_lat, np_ang : float, required. 
            sub-observer longitude, latitude, np_ang in degrees
            see JPL Horizons ephemeris tool for detailed descriptions
        pixscale_km : float, required. pixel scale in km
        req, rpol : float, required. planet equatorial, polar radius in km
        center : 2-element array-like, optional. default (0,0). 
            pixel location of center of planet
        
        Attributes
        ----------
        
        Examples
        --------
        
        '''
        
        # TO DO: handle Astropy units here
        self.ob_lon = ob_lon
        self.ob_lat = ob_lat
        self.np_ang = np_ang
        
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
        '''
        String representation
        
        Examples
        --------
        >>> from pylanetary.planetnav import ModelPlanetEllipsoid
        >>> uranus_model = ModelPlanetEllipsoid(whatever)
        >>> print(uranus_model)
        ModelPlanetEllipsoid instance; req=whatever, rpol=whatever
        '''
        return f'ModelPlanetEllipsoid instance; req={self.req}, rpol={self.rpol}'
        
        
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
        

class PlanetNav(ModelPlanetEllipsoid):
    '''
    use model planet ellipsoid to navigate image data for a planetary body
    
    questions:
        is it possible to take req and rpol from Horizons?
        maybe just have a dict of them
        how should image be passed? as an Image() object?
            do Image() objects end up in utils?
    
    '''
    
    
    def __init__(self, data, ephem, req, rpol, pixscale):
        '''
        Build the planet model according to the observation parameters
        
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
        data : 
        req, rpol : 
        pixscale_arcsec : 
        pixscale_km :
        ephem : 
        deg_per_px : approximate size of pixel on planet, in degrees, at sub-observer point
        lat_g :
        lon_w : 
        mu :
        surf_n :
        
        Examples
        --------
        
        
        '''
        
        # TO DO: fix these all to accept Astropy quantities
        self.data = data
        self.req = req
        self.rpol = rpol
        self.pixscale_arcsec = pixscale
        self.ephem = ephem
        self.pixscale_km = self.ephem['delta']*u.au.to(u.km)*np.radians(self.pixscale_arcsec/3600.)
        
        avg_circumference = 2*np.pi*((self.req + self.rpol)/2.0)
        self.deg_per_px = self.pixscale_km * (1/avg_circumference) * 360 
        
        # build the planet model onto the x-y array of the detector
        super().__init__(self.data.T.shape,self.ephem['PDObsLon'],self.ephem['PDObsLat'],self.pixscale_km,self.ephem['NPole_ang'],self.req,self.rpol)
        
          
    def __str__(self):
        '''
        String representation
        
        Examples
        --------
        >>> from pylanetary.planetnav import PlanetNav
        >>> uranus_obs = PlanetNav(whatever)
        >>> print(uranus_obs)
        PlanetNav instance; req=whatever, rpol=whatever, pixscale=whatever
        '''
        return f'PlanetNav instance; req={self.req}, rpol={self.rpol}, pixscale={self.pixscale}'
    
        
    def ldmodel(self, tb, a, fwhm = 0.0, law='exp'):
        '''
        Make a limb-darkened model disk convolved with the beam
        Parameters
        ----------
        tb: float. brightness temperature of disk at mu=1
        a: float, required. limb darkening parameter
        fwhm: float, optional. FWHM of 1-D gaussian to convolve, units arcsec. 
            if set to 0 (default), does not convolve with beam
        law: str, optional. options 'exp' or 'linear'. type of limb darkening law to use
        '''
        ## TO DO: make this allow a 2-D Gaussian beam!
        
        ldmodel = limb_darkening(np.copy(self.mu), a, law=law)
        ldmodel[np.isnan(ldmodel)] = 0.0
        ldmodel = tb*ldmodel
        if fwhm > 0.0:
            fwhm /= self.pixscale_arcsec #arcsec is assumed, so convert to pixels
            sigma = fwhm / (2*np.sqrt(2*np.log(2))) # convert beam FWHM to 
            model = ndimage.gaussian_filter(ldmodel, sigma) # Gaussian approximation to Airy ring has this sigma
            return model
        elif fwhm == 0:
            return ldmodel
        else:
            raise ValueError("FWHM must be a positive float, or zero (for no beam convolution).")
    
        
    def colocate(self, mode = 'canny', diagnostic_plot=True, save_plot=None, **kwargs):
        '''
        Co-locate the model planet with the observed planet
            using any of several different methods
        
        Parameters
        ----------
        mode : str, optional. Which method should be used to overlay
            planet model and data. Choices are:
            'canny': uses the Canny edge detection algorithm...
                kwargs: low_thresh, high_thresh, sigma
                Question: how to write docstrings for kwargs?
                Question: how to actually do kwargs?
            'manual': shift by a user-defined number of pixels in x and y
                kwargs: shift_x, shift_y
            'convolution': takes the shift that maximizes the convolution of model and planet
                kwargs: {'tb':float, 'a':float, 'fwhm':float}
                    {brightness temperature of disk at mu=1, limb darkening param, fwhm in arcsec}
            another one from skimage tutorial? or this just improves the Canny one?
            default 'canny'
        diagnostic_plot: bool, optional. default True
            do you want the diagnostic plots to be shown
        save_plot: str or None, optional. 
            if str, file path to save the diagnostic plot. if None, does not save.
            does nothing if diagnostic_plot == False
        
        Returns
        -------
            dx, dy:
            dxerr, dyerr:
        
        Examples
        --------
        need at least one example of each mode here
        and one test of each mode, too, btw
        '''
        
        if (mode == 'convolution') or (mode == 'disk'):
            model = self.ldmodel(kwargs['tb'], kwargs['a'], fwhm=kwargs['fwhm'], law='exp')
            data_to_compare = self.data 
        elif mode == 'canny':
            ### COMPLETELY UNTESTED RIGHT NOW ###
            model_planet = ~np.isnan(mu) #flat disk model
            edges = feature.canny(self.data/np.max(self.data), sigma=kwargs['sigma'], low_threshold = kwargs['low_thresh'], high_threshold = kwargs['high_thresh'])
            model = feature.canny(model_planet, sigma=kwargs['sigma'], low_threshold = kwargs['low_thresh'], high_threshold = kwargs['high_thresh'])
            data_to_compare = edges
        [dx,dy,dxerr,dyerr] = chi2_shift(model, data_to_compare)
        
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
        simple function to FFTshift data by a user-defined amount
        for example, to apply the suggested shift from colocate()
        
        Parameters
        ----------
        dx, dy : floats, required. x,y shift in units of pixels
        '''
        
        good = ~np.isnan(self.mu)
        good_shifted = shift2d(good,dx,dy)
        bad_shifted = good_shifted < 0.01
        outputs = []
        for arr in [self.mu, self.lon_w, self.lat_g]:
            
            arr[~good] = 0.0
            arr_shifted = shift2d(arr,dx,dy)
            arr_shifted[bad_shifted] = np.nan
            outputs.append(arr_shifted)
            
        self.mu, self.lon_w, self.lat_g = outputs 
        
        
    def plot_latlon_overlay(self):
        
        return
        
    
    def reproject(self, outstem = None, pixscale_arcsec = None, interp = 'cubic'):
        '''
        Projects the data onto a flat x-y grid according to self.lat_g, self.lon_w
        
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
        Finish this! 
        
        reproject to any geometry, or at least to polar geometry
        '''
        
        #determine the number of pixels in resampled image
        if pixscale_arcsec is None:
            pixscale_arcsec = self.pixscale_arcsec
        npix_per_degree = (1/self.deg_per_px) * (pixscale_arcsec / self.pixscale_arcsec) # (old pixel / degree lat) * (new_pixscale / old_pixscale) = new pixel / degree lat
        npix = int(npix_per_degree * 180) + 1 #(new pixel / degree lat) * (degree lat / planet) = new pixel / planet
        print('New image will be %d by %d pixels'%(2*npix + 1, npix))
        print('Pixel scale %f km = %f pixels per degree'%(self.pixscale_km, npix_per_degree))
        
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
        
        if outstem is not None:
            raise NotImplementedError
            ### need to rewrite this to make fits files from scratch
            ### with some useful header information
            #write data to fits file    
            hdulist_out = self.im.hdulist
            ## projected data
            hdulist_out[0].header['OBJECT'] = self.date+'_projected'
            hdulist_out[0].data = datsort
            hdulist_out[0].writeto(outstem + '_proj.fits', overwrite=True)
            ## emission angles
            hdulist_out[0].header['OBJECT'] = self.date+'_mu_proj'
            hdulist_out[0].data = emang
            hdulist_out[0].writeto(outstem + '_mu_proj.fits', overwrite=True)
            print('Writing files %s'%outstem + '_proj.fits and %s'%outstem + '_mu_proj.fits')

        return projected, mu_projected
        

