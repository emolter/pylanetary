#!/usr/bin/env python
'''

'''


def lat_lon(x,y,ob_lon,ob_lat,pixscale_km,np_ang,req,rpol):
    '''Find latitude and longitude on planet given x,y pixel locations and
    planet equatorial and polar radius'''
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
    #plt.imshow(lon_w, origin = 'lower left')
    #plt.show()
    return lat_g, lat_c, lon_w

    
def surface_normal(lat_g, lon_w, ob_lon):
    '''Returns the normal vector to the surface of the planet.
    Take dot product with sub-obs or sub-sun vector to find cosine of emission angle'''
    nx = np.cos(np.radians(lat_g))*np.cos(np.radians(lon_w-ob_lon))
    ny = np.cos(np.radians(lat_g))*np.sin(np.radians(lon_w-ob_lon))
    nz = np.sin(np.radians(lat_g))
    return np.asarray([nx,ny,nz])


def emission_angle(ob_lat, surf_n):
    '''Return the cosine of the emission angle of surface wrt observer'''
    ob = np.asarray([np.cos(np.radians(ob_lat)),0,np.sin(np.radians(ob_lat))])
    return np.dot(surf_n.T, ob).T
    

class PlanetNav:
    '''
    combine 2-D image data with 
    
    questions:
        is it possible to take req and rpol from Horizons?
        how should image be passed? as an Image() object?
            do Image() objects end up in utils?
    
    ephem: one line from an astroquery.horizons return
        should be ok to pass multiple lines, but will just take the 0th
    '''
    
    
    def __init__(self, image, ephem, req, rpol, pixscale_arcsec):
        '''
        image is an image.Image
        '''
        
        self.image = image
        self.req = req
        self.rpol = rpol
        self.pixscale_arcsec = pixscale_arcsec
        self.ephem = ephem
        self.pixscale_km = self.ephem['delta']*np.radians(self.pixscale_arcsec/3600.)
        
        # build the planet model onto the x-y array of the detector
        (imsize_x, imsize_y) = image.data.shape
        xcen, ycen = int(imsize_x/2), int(imsize_y/2) #pixels at center of planet
        xx = np.arange(imsize_x) - xcen
        yy = np.arange(imsize_y) - ycen
        x,y = np.meshgrid(xx,yy)
        self.lat_g, self.lat_c, self.lon_w = lat_lon(x,y,self.ephem['PDObsLon'],self.ephem['PDObsLat'],self.pixscale_km,self.ephem['NPole_ang'],self.req,self.rpol)
        self.surf_n = surface_normal(self.lat_g, self.lon_w, self.ephem['PDObsLon'])
        self.mu = emission_angle(self.ephem['PDObsLat'], self.surf_n)
    
        
    def edge_detect(self, low_thresh = 0.01, high_thresh = 0.05, sigma = 5, diagnostic_plots = True):
        '''Uses skimage canny algorithm to find edges of planet, correlates
        that with edges of model, '''
        
        
        # this should probably return the ideal shift and apply that shift and plot separately
        # keep the diagnostic plots
        return
        
        
        
    def plot_latlon_overlay(self):
        
        return
        
    
    def project(self):
        
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
        

