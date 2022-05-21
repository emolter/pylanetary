# Licensed under a ??? style license - see LICENSE.rst
import numpy as np


def lat_lon(x,y,ob_lon,ob_lat,pixscale_km,np_ang,req,rpol):
    '''
    Find latitude and longitude on planet given x,y pixel locations and
    planet equatorial and polar radius
    
    Parameters
    ----------
    
    Returns
    -------
    
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
    

class ModelPlanetEllipsoid:
    '''
    Projection of an ellipsoid onto a 2-D array withlatitude and longitude grid
    '''
    
    def __init__(self, shape, ob_lon, ob_lat, pixscale, np_ang, req, rpol):
        '''
        Parameters
        ----------
        
        Attributes
        ----------
        
        Examples
        --------
        
        '''
        
        # TO DO: handle Astropy units here
        self.ob_lon = ob_lon
        self.ob_lat = ob_lat
        ...
        
        xcen, ycen = int(shape[0]/2), int(shape[1]/2) #pixels at center of planet
        xx = np.arange(shape[0]) - xcen
        yy = np.arange(shape[1]) - ycen
        x,y = np.meshgrid(xx,yy)
        self.lat_g, self.lat_c, self.lon_w = lat_lon(x,y,ob_lon,ob_lat,pixscale,np_ang,req,rpol)
        
        # TO DO: test lon_e vs lon_w for different planets!
        # different systems are default for different planets!
        
    def surface_normal(self):
        '''
        Computes the normal vector to the surface of the planet.
        Take dot product with sub-obs or sub-sun vector to find cosine of emission angle
        
        Returns
        -------
        
        Examples
        --------
        
        '''
        nx = np.cos(np.radians(self.lat_g))*np.cos(np.radians(self.lon_w-self.ob_lon))
        ny = np.cos(np.radians(self.lat_g))*np.sin(np.radians(self.lon_w-self.ob_lon))
        nz = np.sin(np.radians(self.lat_g))
        return np.asarray([nx,ny,nz])
        
    def emission_angle(self):
        '''
        Computes the cosine of the emission angle of surface wrt observer
        
        Returns
        -------
        
        Examples
        --------
        
        '''
        surf_n = self.surface_normal()
        ob = np.asarray([np.cos(np.radians(self.ob_lat)),0,np.sin(np.radians(self.ob_lat))])
        return np.dot(surf_n.T, ob).T
        

class PlanetNav(ModelPlanetEllipsoid):
    '''
    use model planet ellipsoid to navigate image data for a planetary body
    
    questions:
        is it possible to take req and rpol from Horizons?
        maybe just have a dict of them
        how should image be passed? as an Image() object?
            do Image() objects end up in utils?
    
    '''
    
    
    def __init__(self, image, ephem, req, rpol, pixscale):
        '''
        Build the planet model according to the observation parameters
        
        Parameters
        ----------
        image : image.Image object, required. 
        ephem : Astropy QTable, required. one line from an astroquery.horizons ephemeris. needs to at least have
            the 'PDObsLon', 'PDObsLat', and 'NPole_ang' fields
            it's ok to pass multiple lines, but will extract relevant fields from the 0th line
        req : float or Quantity object, required. equatorial radius of planet
        rpol : float or Quantity object, required. polar radius of planet
        pixscale : float or Quantity object, required. pixel scale of the images.
        
        Attributes
        ----------
        
        Examples
        --------
        
        
        '''
        
        # TO DO: fix these all to accept Astropy quantities
        self.image = image
        self.req = req
        self.rpol = rpol
        self.pixscale_arcsec = pixscale_arcsec
        self.ephem = ephem
        self.pixscale_km = self.ephem['delta']*np.radians(self.pixscale_arcsec/3600.)
        
        # build the planet model onto the x-y array of the detector
        super.__init__(self.image.shape,self.ephem['PDObsLon'],self.ephem['PDObsLat'],self.pixscale_km,self.ephem['NPole_ang'],self.req,self.rpol)
        self.surf_n = self.surface_normal()
        self.mu = self.emission_angle()
    
        
    def overlay_model(self, mode = 'canny', **kwargs):
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
                kwargs: model_brightness
            another one from skimage tutorial? or this just improves the Canny one?
            default 'canny'
        
        Returns
        -------
        
        Examples
        --------
        need at least one example of each mode here
        and one test of each mode, too, btw
        '''
        
        
        # this should probably return the ideal shift and apply that shift and plot separately
        # keep the diagnostic plots
        return
        
        
        
    def plot_latlon_overlay(self):
        
        return
        
    
    def reproject(self):
        
        # would be cool to be able to project to any geometry
        # need Chris here
        return
    
    
    def write(self, target, lead_string):
        '''Tertiary data products'''
        hdulist_out = self.im.hdulist
        #centered data
        hdulist_out[0].header['OBJECT'] = target+'_CENTERED'
        hdulist_out[0].data = self.centered
        hdulist_out[0].writeto(lead_string + '_centered.fits', overwrite=True)
        #latitudes
        hdulist_out[0].header['OBJECT'] = target+'_LATITUDES'
        hdulist_out[0].data = self.lat_g
        hdulist_out[0].writeto(lead_string + '_latg.fits', overwrite=True)
        #longitudes
        hdulist_out[0].header['OBJECT'] = target+'_LONGITUDES'
        hdulist_out[0].data = self.lon_w
        hdulist_out[0].writeto(lead_string + '_lonw.fits', overwrite=True)
        

