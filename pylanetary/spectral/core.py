import os,sys
import numpy as np
from scipy.stats import scoreatpercentile
import pickle
from lmfit.models import GaussianModel,LorentzianModel,VoigtModel,PseudoVoigtModel,MoffatModel
from lmfit.model import save_modelresult,load_modelresult
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import scipy.io
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.constants import c
#from pylanetary.utils import *
import pdb

def centered_list(n):
    if n % 2 == 0: # If n is even, we adjust it to the next odd number
        n += 1
    half = n // 2
    centered_list = list(range(-half, half + 1,2))
    return centered_list

def calc_doppler_vel(peak, err, rest_f):
    '''
    Calculate the Doppler velocity from the calculated center frequency.
    
    Parameters
    ---------- 
    peak - the calculated central peak frequency of the spectral line peak, float
    err - the standard deviation of the calculated central peak frequency, float
    rest_f - the rest frequency of the transition, float
    
    Returns
    -------
    v - the calculated Doppler velocity, m/s.
    wind_err - the Doppler velocity error, calculated through error propagation, m/s.
    '''
    a = c.value/rest_f
    # Doppler equation
    v = ((-1*peak/rest_f)+1)*c.value # Calculate Doppler shifted winds
    print('The wind speed is '+str(v)+'m/s')
    wind_err = np.abs(a*err) # Calculate the error in velocity
    print('The wind error is '+str(wind_err)+'m/s')
    
    return v, wind_err
    
    
# Guide to keys in the nested model dictionary:
# model[] - Model name, taken from user input, string
# ref - short reference name of model type, string
# lmfit_func - built-in LMFIT function name, text 
# label - fit label for plotting - string
    
model = {}
model['Gaussian'] = {'ref': 'gauss', 'lmfit_func': GaussianModel(),'label': 'Gaussian Fit','color':'orange'}
model['Lorentzian'] = {'ref': 'lor','lmfit_func':LorentzianModel(),'label': 'Lorentzian Fit','color':'green'}
model['Voigt'] = {'ref': 'voigt','lmfit_func':VoigtModel(),'label': 'Voigt Fit','color':'red'}
model['PseudoV'] = {'ref': 'psv','lmfit_func':PseudoVoigtModel(),'label': 'Pseudo-Voigt Fit','color':'purple'}
model['Moffat'] = {'ref': 'mof','lmfit_func':MoffatModel(),'label': 'Moffat Fit','color':'brown'}

###############################################################################
###############################################################################

class Spectrum:
  '''
  Examples
  --------
  Initialize a Spectrum object:
  your_spec = Spectrum(spec, freq = freq,rest_f = 345.79598990,RMS = 0.017530593)
  
  Calculate the best fit of the spectrum using a Lorentzian fit:
  test_peak, test_std, chi, chi_red = test.fit_profile(fit_type = 'Lorentzian')
  
  Calculate the best fit of the spectrum with all line shapes and save the figure:
  test_peak, test_std, chi, chi_red = test.fit_profile(fit_type = 'Moffat', plot_name = 'line_comparison.png')
  
  Calculate the Doppler shifted winds:
  test_wind, test_err = test.calc_doppler_vel(peak = test_peak, err = test_std)
  '''
  
  ###############################################################################
  
  def __init__(self, spec, rest_f, x_space, wl = None, freq = None, RMS = None):
    '''
    Parameters
    ----------
    spec = intensity axis, numpy array of floats (required)
    rest_f = rest frequency of molecular transition, float (required for fit_profile and calc_doppler_vel) 
    x_space - identifier for choosing wavelength or frequency space for the model fit, string. Input 'F' or 'f' for frequency space, 'W' or 'w' for wavelength space. 
    wl = wavelength axis, optional. One of wl or freq must be specified
    freq = frequency axis, optional. One of wl or freq must be specified.
    RMS (optional) - numerical value of the data's RMS, float. If not specified, an RMS value is estimated through sigma clipping.
    
    Calculated Attributes
    ---------------------
    xaxis - the frequency axis used for all calculations. Determined by the x_space parameter.
    wl_res - wavelength resoltuion
    freq_res - frequency resolution
    chan_num - number of wavelength/frequency channels
    
    '''
    self.wl = wl
    self.freq = freq

    if self.wl is None:
      self.wl = c/self.freq
    elif self.freq is None:
      self.freq = c/self.wl
    
    if x_space == 'F' or x_space == 'f':
      self.xaxis = self.freq
    elif x_space == 'W' or x_space == 'w':
      self.xaxis = self.wl_spec
    else:
      raise ValueError('You did not choose frequency or wavelength space. Please input F/f for frequency, W/w for wavelength.')

    self.spec = spec
    self.rest_f = rest_f
    
    if RMS is None:
      self.RMS,median,rms_std = sigma_clipped_stats(self.spec)
    else:
      self.RMS = RMS
      
    self.wl_res = self.wl[1] - self.wl[0]
    self.freq_res = self.freq[1] - self.freq[0]
    self.chan_num = np.size(self.wl)
	
  ###############################################################################
  
  def make_initial_guess(self, fit_type, outfile):
    '''
    A LMFIT wrapper to save a model fit as an initial guess. This can improve the final results when fitting an image cube.
    
    Parameters
    ----------
    fit_type - the type of line shape used to generate the initial guess. the 'All' fit type is not allowed and will result in an error.
    outfile - name of the output file, string.
    
    '''
    
    pars = model[fit_type]['lmfit_func'].guess(self.spec,x=self.xaxis)
    fit_result = model[fit_type]['lmfit_func'].fit(self.spec,pars,x=self.xaxis,scale_covar=False)
    print(fit_result.fit_report())
    save_modelresult(fit_result, outfile+'.sav')
    
  ###############################################################################
    
  def fit_profile(self, fit_type, showplots = True, plot_name = None, initial_fit_guess = None):
    '''
    Calculate the best fit of a Spectral object using the LMFIT algorithm. The line shapes can be plotted for visual comparison.
    
    Parameters
    ----------
    fit_type - the type of line shape used to fit the spectra, string. Options are:
        All: Fit the spectrum with all line 5 line shape options. Data is always plotted.
        Gaussian: fit the spectrum with a Gaussian function
        Lorentzian: fit the spectrum with a Lorentzian function
        Voigt: fit the spectrum with a Voigt function
        PseudoV: fit the spectrum with a Pseudo-Voigt function
        Moffat: fit the spectrum with a Moffat function
        A description of all line shapes can be found at https://lmfit.github.io/lmfit-py/builtin_models.html
        
    showplots - indicator of whether to plot. Default is True.
    plot_name - name of the saved figure, optional. Default is None.
    initial_fit_guess (optional) - path to a saved LMFIT result to use as the initial guess. This can be beneficial if some spectra are very noisy. It is recommended that you use a pixel with as small a Doppler shift as possible to make your initial guess. Default is None.
    
    Returns
    -------
    peak - frequency or wavelength value of the line peak, LMFIT's 'center' parameter.
    err - standard deviation of the frequency or wavelength value.
    chi - calculated Chi-squared statistic of the line fit.
    chi_red - calculated reduced Chi-squared statistic of the line fit.
    '''
    
    if fit_type != 'All':
      if initial_fit_guess == None: # If no initial guess is given, calculate one using the built-in LMFIT .guess method
        pars = model[fit_type]['lmfit_func'].guess(self.spec,x=self.xaxis)
      else: # Load initial guess if parameter is provided
        inital = load_modelresult(initial_fit_guess) # Inital guess that is best fit from center pixel
        pars = inital.params
      fit_result = model[fit_type]['lmfit_func'].fit(self.spec,pars,x=self.xaxis,scale_covar=False)
      print(fit_result.fit_report()) # Print out the best fit parameters and statistics
    elif fit_type == 'All':
      fit_result = {}
      for key in model:
        pars = model[key]['lmfit_func'].guess(self.spec,x=self.xaxis)
        result = model[key]['lmfit_func'].fit(self.spec,pars,x=self.xaxis,scale_covar=False)
        print(result.fit_report())
        fit_result[key] = result
    else:
      raise ValueError('fit_type must be one of the following options: Gaussian, Lorentzian, Voigt, PseudoV, Moffat, or All')
    
    # Calculate statistics manually for single fit type
    if fit_type != 'All':
      chi = np.sum(((np.subtract(self.spec,fit_result.best_fit)/self.RMS))**2)
      chi_red = chi/(self.chan_num - 4)
    else:
      chi = np.NAN
      chi_red = np.NAN

    if showplots is True:
      test.plot_lines(line_result = fit_result, model_key = fit_type)
    
    if plot_name is not None:
      plt.savefig(plot_name)
    
    if fit_type != 'All':
      peak = fit_result.params['center']
      err = fit_result.params['center'].stderr
    else:
      peak = 0
      err = 0
    
    return peak, err, chi, chi_red # think about what happens if lmfit results are unreasonable… but this should probably happen elsewhere
    
  ############################################################################### 
  
  def plot_lines(self, line_result, model_key, x_label = 'Frequency (GHz)', y_label = 'Intensity (mJy)', residual_label = 'Residuals', plot_title = 'Comparison of Line Shapes'):
    '''
    Plot the original data and best fit line shape for visual comparison.
    
    Parameters
    ----------
    line_result - the LMFIT best fit result.
    model_key - the name of the line shape used to calculate the fit.
    
    x_label - x axis label for the main and residual plot, optional. Default is 'Frequency (GHz)'.
    y_label - y axis label for the main plot, optional. Default is 'Intensity (mJy)'.
    residual_label - y axis label for the residual plot, optional. Default is 'Residuals'.
    plot_title - title for the plot, optional. Default is 'Comparison of Line shapes'.
    
    '''
    
    f, (a0,a1) = plt.subplots(2,1,height_ratios = [3,1])
    a0.vlines(x = self.rest_f, ymin=0, ymax=np.max(self.spec), color = 'grey', linestyles='dashed', label='Rest Frequency')
    a0.errorbar(self.xaxis, self.spec, yerr=self.RMS, fmt='o', label='Original Data')
    
    if model_key == 'All':
      for key in model:
        residuals = self.spec - line_result[key].best_fit
        a0.plot(self.xaxis,line_result[key].best_fit,'-',label = model[key]['label'],color = model[key]['color'])
        a1.plot(self.xaxis,residuals,'--',color = model[key]['color'])
    else:
      residuals = self.spec - line_result.best_fit
      a0.plot(self.xaxis,line_result.best_fit,'-',label = model[model_key]['label'],color = model[model_key]['color'])
      a1.plot(self.xaxis,residuals,'--',color = model[model_key]['color'])
    
    a1.set_xlabel(x_label)
    a0.set_ylabel(y_label)
    a1.set_ylabel(residual_label)
    a0.title.set_text(plot_title)
    a0.legend()
    plt.show()
    
  ###############################################################################
'''   
# Simple single spectra testing case
if __name__=="__main__":
  
  # CO [10,25]
  
  freq = np.array([345.76770566,345.76868215,345.76965863,345.77063511,345.77161159,345.77258807,345.77356455,345.77454103,345.77551751,345.77649399,345.77747047,345.77844695,345.77942343,345.78039991,345.78137639,345.78235287,345.78332935,345.78430584,345.78528232,345.7862588,345.78723528,345.78821176,345.78918824,345.79016472,345.7911412,345.79211768,345.79309416,345.79407064,345.79504712,345.7960236,345.79700008,345.79797656,345.79895304,345.79992953,345.80090601,345.80188249,345.80285897,345.80383545,345.80481193,345.80578841,345.80676489,345.80774137,345.80871785,345.80969433,345.81067081,345.81164729,345.81262377,345.81360025,345.81457673,345.81555322,345.8165297,345.81750618,345.81848266,345.81945914,345.82043562,345.8214121,345.82238858])
  spec = np.array([0.21198112,0.18924561,0.21650964,0.22294408,0.24960007,0.22984277,0.25219446,0.24985315,0.25343025,0.28705364,0.3061584,0.31352046,0.3277193,0.3265509,0.3436137,0.35133913,0.38450843,0.4165357,0.43355167,0.467518,0.49524495,0.5277506,0.5910206,0.64392966,0.7271694,0.89640355,0.8984462,0.76777405,0.6531943,0.59134954,0.55332845,0.50916755,0.46389887,0.4410508,0.4036594,0.40885097,0.37320843,0.3508784,0.34755984,0.31414196,0.29820374,0.27975425,0.28136522,0.26828533,0.26605716,0.24770543,0.2537956,0.22927019,0.219696,0.21160068,0.20004132,0.1939475,0.18222019,0.16944197,0.17536835,0.16506103,0.16306329])
  
  test = Spectrum(spec, freq = freq,rest_f = 345.79598990,x_space = 'f', RMS = 0.017530593)
  #test = Spectrum(spec, freq = freq) #HCN
  #test.make_initial_guess(fit_type = 'Moffat', outfile = 'moffat_guess')
  test_peak, test_std, chi, chi_red = test.fit_profile(fit_type = 'Moffat', initial_fit_guess = 'moffat_guess.sav')
  test_wind, test_err = calc_doppler_vel(peak = test_peak, err = test_std, rest_f = 345.79598990)
  #test_wind, test_err = test.calc_doppler_vel(rest_freq = 354.50547790, peak = test_peak, err = test_std) # HCN
'''

###############################################################################
###############################################################################
  
class SpectralCube:
  '''
  Examples
  --------
  Initilaize the class:
  cube = SpectralCube(your_image_path)
  
  Load an entire image cube in frequency space:
  
  Load a single row in wavelength space:
  
  Load a single pixel in frequency space:
  
  Example Workflow
  ----------------
  
  '''
  ###############################################################################
  
  def __init__(self, imagepath, x_space):
    '''
    Parameters
    ----------
    imagepath - path to FITS image file, string (required)
    x_space - identifier for choosing wavelength or frequency space for the model fit, string. Input 'F' or 'f' for frequency space, 'W' or 'w' for wavelength space.
    
    Attributes
    ----------
    chan_num - number of frequency channels
    i0 - reference pixel
    df - channel width (Hz)
    f0 - rest frequency (Hz)
    xpixmax - x-dimension size
    ypixmax - y-dimension size
    bmaj - semi-major axis of the restoring beam, arcsec. 
    bmin - semi-minor axis of the restoring beam, arcsec.
    bpa - position angle of the restoring beam, degrees.
    data - spectral data
    freqspec - frequency grid around the rest frequency
    wl_spec - wavelength grid around the rest wavelength
    
    '''  
    self.imagepath = imagepath
    self.x_space = x_space
    
    fits_file = fits.open(self.imagepath)
    hdr = fits_file[0].header
    #print(hdr)
    #breakpoint()
    # Pull necessary info from the FITS header
    self.chan_num = hdr['NAXIS3'] 
    self.i0 = hdr['CRPIX3'] 
    self.df = hdr['CDELT3'] 
    self.f0 = hdr['CRVAL3']
    self.xpixmax = hdr['NAXIS1'] 
    self.ypixmax = hdr['NAXIS2']
    self.bmaj = hdr['BMAJ']
    self.bmin = hdr['BMIN']
    self.bpa = hdr['BPA']
    self.data = fits_file[0].data
    fits_file.close()
    
    print('Number of pixels: '+str(self.xpixmax)+','+str(self.ypixmax))
    # Correct for FITS 1-indexing scheme
    if self.i0 < 0: 
      self.i0 = self.i0 - 1
    else:
      self.i0 = self.i0 + 1
    
    self.freqspec = ((np.arange(self.chan_num) - self.i0) * self.df + self.f0)/1e9
    self.wl_spec = c/self.freqspec
      
    if x_space == 'F' or x_space == 'f':
      self.xaxis = self.freqspec
    elif x_space == 'W' or x_space == 'w':
      self.xaxis = self.wl_spec
    else:
      raise ValueError('You did not choose frequency or wavelength space. Please input F/f for frequency, W/w for wavelength.')
      
  ###############################################################################
  
  def extract_pixel(self, pixel):
    '''
    Extract the spectral axis for a single pixel
    
    Parameters
    ----------
    pixel - indexes of the pixel to be fitted, list.
    
    Returns
    -------
    datas - numpy array containing spectral axis
    
    '''
    print('Fitting pixel ['+str(pixel[0])+','+str(pixel[1])+']')
    datas = np.empty((1,1),dtype=np.ndarray)
    
    print('Extracting spectrum at pixel '+str(pixel[0])+','+str(pixel[1]))
    datas = self.data[0,:,pixel[1],pixel[0]]
    
    return datas
  
  ###############################################################################
    
  def extract_image(self):
    '''
    Extract the spectral axis for the entire image cube
    
    Returns
    -------
    datas - 3D numpy array containing the spectral axis
    '''
    
    print('Fitting the entire image')
    datas = np.empty((self.xpixmax,self.ypixmax),dtype=np.ndarray)
    
    for xpix in range (0,self.xpixmax,1):
      for ypix in range (0,self.ypixmax,1):
        print('Extracting spectrum at pixel '+str(xpix)+','+str(ypix))
        datas[xpix,ypix] = self.data[0,:,ypix,xpix]
    
    return datas
 
  ###############################################################################
  
  def make_mask(self,pixels):
    '''
    Make a boolean array to act as a mask for extracting pixels
    
    Parameters
    ----------
    pixels - tuple of pixel values to include in the mask. These pixels will be given a value of True.
    
    Returns
    -------
    mask - the boolean array mask
    
    '''
    
    mask = np.full((self.xpixmax,self.ypixmax), 0)
    for i in pixels:
      ix = i[0]
      iy = i[1]
      mask[ix,iy] = 1
    mask = np.array(mask,dtype='bool')
      
    return mask
    
  ###############################################################################  
    
  def extract_mask_region(self,mask):
    '''
    Extract pixels defined in a boolean array
    
    Parameters
    ----------
    mask - 2D boolean array of the size [xpixmax,ypixmax]. Pixels with a value of True are extracted for fitting.
    
    Returns
    -------
    datas - 3D numpy array containing the spectral axis. Masked pixels are NAN. 
    
    '''
    datas = np.empty((self.xpixmax,self.ypixmax),dtype=np.ndarray)
    
    for xpix in range (0,self.xpixmax,1):
      for ypix in range (0,self.ypixmax,1):
      
        if mask[xpix,ypix] == True:
          datas[xpix,ypix] = self.data[0,:,ypix,xpix]
        else:
          datas[xpix,ypix] = np.NAN
          
    return datas
  
  
  ###############################################################################
  
  def fit_data(self, datas, fit_type, outfile, initial_fit_guess = None, RMS = None, SN = None):
    '''
    Parameters
    ----------
    datas - array of spectral data to be fit. Use the datas output from the load_data method.
    fit_type - the type of line shape used to fit the spectra, string.
    outfile - name of the saved output pickle file, string. The 'pickle' file type is automatically appied, no need to include it here.

    initial_fit_guess (optional) - a saved LMFIT result to use instead of the built in guess function, string. This can be beneficial if some spectra are very noisy. If desired, it is recommended that you save the model result of the image's central pixel to use as the inital guess. See LMFIT documentation (https://lmfit.github.io/lmfit-py/model.html#saving-and-loading-modelresults) 
    RMS (optional) - numerical value of the data's RMS, float. If not provided, an estimate is calculated using Astropy's siga_clipped_stats
    SN (optional) - Signal to noise ratio minimum, float. Default is value is 1. This is used to mask out bad data; any pixel witha S/N lower than this value will not be fit.
    
    '''
    
    if SN == None:
      SN = 1
    
    # Initialize output file
    picklefile = open(outfile+'.pickle', 'wb')
    print('Numerical results will be saved in the '+outfile+'.pickle file')
    results = {'x':[],'y':[],'center':[],'std':[],'chi':[],'redchi':[]} # Define output dictionary fields
    
    
    #for (xpix,ypix), value in np.ndenumerate(datas):
    for xpix in range (0,self.xpixmax,1):
      for ypix in range (0,self.ypixmax,1): 
        #breakpoint()
        s_n = abs(np.max(datas[xpix,ypix])/RMS)
        
        if s_n > SN:
          print('Modeling spectrum at pixel '+str(xpix)+','+str(ypix))
          spectra = Spectrum(spec = datas[xpix,ypix], freq = self.xaxis, rest_f = 345.79598990, x_space = self.x_space, RMS = RMS) 
          peak, std, chi, chi_red = spectra.fit_profile(fit_type = 'Moffat', initial_fit_guess = initial_fit_guess, showplots = False)
          # Save output in output dictionary
          results['x'].append(xpix)
          results['y'].append(ypix)
          results['center'].append(peak)
          results['std'].append(std) 
          results['chi'].append(chi)
          results['redchi'].append(chi_red)
        elif s_n <= SN or np.isnan(s_n): 
          results['x'].append(xpix)
          results['y'].append(ypix)
          results['center'].append(np.NAN)
          results['std'].append(np.NAN) 
          results['chi'].append(np.NAN)
          results['redchi'].append(np.NAN)
          
          
    # Save the model results in a pickle file
    lenx = np.shape(datas)[0]
    leny = np.shape(datas)[1]
    #breakpoint()
    results['x']=np.reshape(results['x'],(lenx,leny))
    results['y']=np.reshape(results['y'],(lenx,leny))
    results['center']=np.reshape(results['center'],(lenx,leny))
    results['std']=np.reshape(results['std'],(lenx,leny))
    results['chi']=np.reshape(results['chi'],(lenx,leny))
    results['redchi']=np.reshape(results['redchi'],(lenx,leny))
      
    pickle.dump(results,picklefile)
    
    picklefile.close()
    
  ###############################################################################  
  
  def wind_calc_new(self, picklefile, restfreq, outfile):
    '''
    Parameters
    ----------
    picklefile - path to the pickle file of calculated doppler shifts, string
    restfreq - the rest frequency of the molecular transition, float
    outfile - the name/path of the wind pickle file, string
    
    '''
    
    r = pickle.load(open(picklefile,"rb"),encoding='latin-1') 
    peaks = np.array(r['center'])
    std = r['std']
    
    x_max = peaks.shape[0]
    y_max = peaks.shape[1]
    
    new_pickle = open(outfile+'.pickle', 'wb')
    print('Numerical wind results will be saved in the '+outfile+'.pickle file')
    results = {'x':[],'y':[],'v':[],'v_err':[]} # Define output dictionary fields
    
    for xpix in range (0,x_max,1):
      for ypix in range (0,y_max,1):
        v, wind_err = calc_doppler_vel(peaks[xpix,ypix], std[xpix,ypix], restfreq)
        
        results['x'].append(xpix)
        results['y'].append(ypix)
        results['v'].append(v)
        results['v_err'].append(wind_err)
    
    lenx = np.shape(r['center'])[0]
    leny = np.shape(r['center'])[1]
    
    results['x']=np.reshape(results['x'],(lenx,leny))
    results['y']=np.reshape(results['y'],(lenx,leny))  
    results['v']=np.reshape(results['v'],(lenx,leny))
    results['v_err']=np.reshape(results['v_err'],(lenx,leny))  
    
    pickle.dump(results,new_pickle)
    
  ###############################################################################  
   
  def wind_calc(self, picklefile, restfreq, outfile):
  
    '''
    Parameters
    ----------
    picklefile - path to the pickle file of calculated doppler shifts, string
    restfreq - the rest frequency of the molecular transition, float
    outfile - the name/path of the wind pickle file, string
    
    '''
    r = pickle.load(open(picklefile,"rb"),encoding='latin-1') 
    peaks = np.array(r['center'])
    std = r['std']
    a = c/restfreq # Ratio for calculating errors through error propagation
    
    # Load the pickle file
    picklefile = open(outfile+'.pickle', 'wb')
    print('Numerical wind results will be saved in the '+outfile+'.pickle file')
    results = {'x':[],'y':[],'v':[],'v_err':[]} # Define output dictionary fields
    
    if peaks.ndim == 1: # If the pickle file has one entry (ie. a single pixel)
      x = peaks
      v = ((-1*x/restfreq)+1)*c # Calculate Doppler shifted winds
      print('The wind speed is '+str(v)+'m/s')
      err = np.abs(a*np.array(std)) # Calculate the error in velocity
      print('The wind error is '+str(err)+'m/s')
      results['x'].append(r['x'])
      results['y'].append(r['y'])
      results['v'].append(v)
      results['v_err'].append(err)
    else: # Pickle file contains a row, column, or full image
      for (xpix,ypix), value in np.ndenumerate(peaks):
        x = peaks[xpix,ypix]
        v = ((-1*x/restfreq)+1)*c # Calculate Doppler shifted winds
        print('The wind speed is '+str(v)+'m/s')
        err = np.abs(a*std[xpix,ypix]) # Calculate the error in velocity through error propagation
        print('The wind error is '+str(err)+'m/s')
        results['x'].append(xpix)
        results['y'].append(ypix)
        results['v'].append(v)
        results['v_err'].append(err)
        
      # Save the model results in an updated pickle file
      lenx = np.shape(r['center'])[0]
      leny = np.shape(r['center'])[1]
      
      results['x']=np.reshape(results['x'],(lenx,leny))
      results['y']=np.reshape(results['y'],(lenx,leny))  
      results['v']=np.reshape(results['v'],(lenx,leny))
      results['v_err']=np.reshape(results['v_err'],(lenx,leny))  
    
      pickle.dump(results,picklefile)
      
  ###############################################################################
  
  def plot_data(self, picklefile, platescale, object, title, cont_label, date = None, location = None, spatial = None, variable = None, cmap = None, limits = None, contours = None, hatch = None, savefig = None):
   #spatial, dist, planetrad, subobslat, ccw, bmaj, bmin, bpa, title, cont_label, variable = None, cmap = None, limits = None, contours = None, hatch = None, savefig = None):
  
    '''
    Plot the calculated winds on a 2D plane. 
    This algorithm was originally written by Dr. Martin Cordiner for use on Titan (Cordiner et al. 2020).
    
    Parameters
    ----------
    picklefile - path to the pickle file of calculated doppler winds, string
    platescale - ,float
    object - name of the planetary object you are studying, string.
    title - plot title, string
    cont_label - label for the contour bar, string
    
    date - date of the observation in 'YYYY-MM-DD 00:00' format, optional. Default is the current date.
    location - location of the observing device, default is None. Options include ALMA, VLA, ?
    spatial - numerical value for spatial axes bounds in km, float. If not specified, the value is selected so the image is 1.5 planet diameters wide.
    variable (optional) - name of the variable plotted, string. Default is the Doppler velocities. Options include 'v' and 'v_err'.
    cmap (optional) - name of Matplotlib colormap for plotting. Default is blue/white/red gradient. 
    limits (optional) - limits of the wind colorbar, list. Default is 0 to 1.
    contours (optional) - numerical info for contour lines, list. First value is the number of lines to plot. Second value is the multiplicative factor for the contor values. Example: for [13,200] 13 contour lines between -2200 and 2200 m/s are plotted. Default is [11,10] (11 contour lines from -100 to 100)
    hatch (optional) - boolean for if the plotted beam is hatched or not. Default is False (no hatch marks)
    savefig (optional) - file path for the saved figure, string. The figure is not saved if this parameter is None.
    
    '''
    
    # Define planet parameters with Pylanetary Body class
    planet = Body('object')
    dist = planet.semi_major_axis
    planetrad = planet.req
    subobslat = planet.ephem['PDObsLat']
    ccw = planet.ephem['NPole_ang']
    
    # Define basic plotting parameters
    matplotlib.rcParams["contour.negative_linestyle"]='solid'
    planetgridcolor='black' # Planet disc color
    kmscale=725.27*dist*platescale # image pixel size in km
    
    if spatial is None:
      spatial = planetrad*1.25
    
    #Define the colorbar color map
    if cmap == None: 
      cmap = LinearSegmentedColormap.from_list('mycmap', (['blue','white','red']))
    else:
      cmap = cmap
    # Define the limits of the colorbar. Defualt is 0 to 1.
    if limits == None:
      norm = None
    else:
      norm = colors.Normalize(vmin=limits[0],vmax=limits[1])
      
    # Define the plotted contour levels. Default is 11 contour lines from -100 to 100
    if contours == None:
      levels = np.array(centered_list[10]) * 10
    else:
      levels = np.array(centered_list(contours[0])) * contours[1]
    
    # Load the data
    r = pickle.load(open(picklefile,"rb"),encoding='latin-1') # Load the pickle file with calculated winds
    if variable == None:
      data = r['v']
    else:
      data = r[variable]
    
    subobslat = np.deg2rad(subobslat)
    ccw = np.deg2rad(ccw)
    
     # Define spatial axes
    x = ((r['x']-len(r['x'])/2.))*kmscale
    y = (r['y']-len(r['y'])/2.)*kmscale
    
    # Initialize the figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # Plot the data
    c=ax.imshow(data.transpose(),extent=[x.min(), x.max(), y.min(), y.max()],origin='lower',interpolation='nearest',norm=norm,zorder=-21,cmap=cmap)
    co=ax.contour(data.transpose(),levels=levels,extent=[x.min(), x.max(), y.min(), y.max()],origin='lower',colors='k',linewidths=0.75,zorder=-20)
    # Add labels and define text sizes
    ax.clabel(co,fmt="%d",fontsize=13) # Contour label text size
    ax.set_xlabel('Distance (km)', fontsize = 20)
    ax.set_ylabel('Distance (km)', fontsize = 20)
    ax.tick_params(axis='both',labelsize=18)
    ax.set_title(title, fontsize = 24)
    cb = fig.colorbar(c)
    cb.set_label(cont_label, fontsize = 20)
    cb.ax.tick_params(labelsize = 20)
    
    # Add the beam ellipse
    width = bmin*725.27*dist
    height = bmaj*725.27*dist
    beam = Ellipse(xy=[-3.0e4,-3.2e4], width=bmin*725.27*dist, height=bmaj*725.27*dist, angle=bpa)
    beam.set_facecolor('none')
    beam.set_edgecolor('black')
    if hatch == True: 
      beam.set_hatch('/')
    else:
      pass
    ax.add_artist(beam)
    
    # Add the planet circle and lat/long lines
    planetcircle = Circle((0.0,0.0), planetrad,color=planetgridcolor,linewidth=1.5,zorder=0,alpha=0.5)
    planetcircle.set_facecolor('none')
    ax.add_artist(planetcircle)
    
    #Latitude lines
    latlevel = [-67.5,-45.0,-22.5,0.0,22.5,45.0,67.5]
    grid = np.linspace(-spatial,spatial,1024)
    x,y = np.meshgrid(grid,grid)
    z = np.hypot(x, y)
    northx = -np.sin(ccw)*np.cos(subobslat)
    northy = np.cos(ccw)*np.cos(subobslat)
    northz = np.sin(subobslat) 
    with np.errstate(divide='ignore',invalid='ignore'): 
        zcoord = np.sqrt((planetrad)**2 - x**2 - y**2)
        dprod = (northx*x + northy*y + northz*zcoord)/(planetrad) #dot product of north pole vector and each vector in model planet
        z_lat = 90. - np.degrees(np.arccos(dprod)) #latitude of each point on the 2-d grid
    ax.contour(z_lat,colors=planetgridcolor,extent=[-spatial, spatial, -spatial, spatial],linestyles='dashed',zorder=0,alpha=0.5,levels=latlevel)
    ax.contour(z_lat,colors=planetgridcolor,extent=[-spatial, spatial, -spatial, spatial],levels=[0],linestyles='solid',zorder=0)

    #Longitude lines
    xma=np.ma.masked_where(np.isnan(zcoord),x)
    yma=np.ma.masked_where(np.isnan(zcoord),y)
    # Rotate the x,y coordinates
    xr = np.cos(ccw)*xma + np.sin(ccw)*yma
    yr = -np.sin(ccw)*xma + np.cos(ccw)*yma
    # Rotate the z coordinate
    projz = (zcoord - (yr * np.tan(subobslat))) * np.cos(subobslat)
    z_long = np.rad2deg(np.arctan2(xr,projz)) #longitude
    # Projected distances of each x,z point from the polar vector
    projx = (xma - (-yma * np.tan(ccw))) * np.cos(ccw) 
    projz = (zcoord - (yma * np.tan(subobslat))) * np.cos(subobslat)
    ax.contour(z_long,12,colors=planetgridcolor,extent=[-spatial, spatial, -spatial, spatial],linestyles='dotted',zorder=0)
    
    plt.show()
    if savefig == None:
      print('Figure was not saved')
    else:
      fig.savefig(savefig.png, bbox_inches='tight')
      
  ###############################################################################
    
# Testing case for image cube analysis

if __name__=="__main__":
  image = '/homes/metogra/skathryn/Research/Data/ContSub/CO/CS20/Neptune_Pri_X50a4_CS20_narrow_square_2.fits'
  #image = '/homes/metogra/skathryn/Research/Data/ContSub/HCN/CS12/Neptune_Pri_X50a4_HCN_CS12_narrow_square_2.fits'
  test_cube = SpectralCube(image,x_space = 'f')
  #datas = test_cube.extract_pixel([8,25])
  #datas = test_cube.extract_image()
  mask_pix = [[15,17],[16,17],[17,17],[15,16],[16,16],[17,16],[15,15],[16,15],[17,15]]
  #mask_pix = [[0,0],[1,1]]
  mask = test_cube.make_mask(mask_pix)
  data = test_cube.extract_mask_region(mask)
  #data = test_cube.extract_image()
  
  test_cube.fit_data(data, fit_type = 'Moffat', outfile = 'development_testing_wind', RMS = 0.017530593, SN = 6,initial_fit_guess = '/homes/metogra/skathryn/Research/Scripts/co_moffat_modelresult.sav')
  
  test_cube.wind_calc_new(picklefile = 'development_testing_wind.pickle', restfreq = 345.79598990, outfile = 'development_testing_wind_calc')
  
  #test_cube.fit_cube(datas = test_data, fit_type = 'Moffat', xaxis = test_axis, outfile = 'development_testing_wind', RMS = 0.017530593, initial_fit_guess = '/homes/metogra/skathryn/Research/Scripts/co_moffat_modelresult.sav',SN = 6)
  
  #test_cube.wind_calc(picklefile = '/homes/metogra/skathryn/pylanetary_dev/pylanetary/pylanetary/spectral/development_testing_wind.pickle', restfreq = 345.79598990, outfile = 'development_testing_wind')
  
  #test_cube.plot_data(picklefile = 'development_testing_wind.pickle', platescale = 0.1, spatial = 4e4, dist = 30.4627707075422, planetrad = 24622., subobslat = -26.167088, ccw = -34., bmaj = 0.4416242241859436, bmin = 0.3885720372200012, bpa = 81.32388305664062, title = 'CO Doppler Velocity Map (4/30/16)', cont_label = 'Radial Velocity (m/s)', limits = [-2000,2000], contours = [21,200])
  
