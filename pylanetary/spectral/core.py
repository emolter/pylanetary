import os,sys
import numpy as np
from scipy.stats import scoreatpercentile
import pickle
from lmfit.models import GaussianModel,LorentzianModel,VoigtModel,PseudoVoigtModel,MoffatModel
from lmfit.model import save_modelresult,load_modelresult
from lmfit import Parameters
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import scipy.io
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.constants import c
from pylanetary.utils import *
import time
import pdb

def centered_list(n):
    if n % 2 == 0: # If n is even, we adjust it to the next odd number
        n += 1
    half = n // 2
    centered_list = list(range(-half, half + 1,2))
    return centered_list

def lmfit_wrapper(fit_type, spec, xaxis, RMS, initial_fit_guess = None, weight_range = None):
  '''
  Function to interface with LMFIT and calculate a model fit.
  
  Parameters
  ----------
  fit_type : string
        The type of line shape used to fit the spectra, string. Options are:
          All: Fit the spectrum with all line 5 line shape options. Data is always plotted.
          Gaussian: fit the spectrum with a Gaussian function
          Lorentzian: fit the spectrum with a Lorentzian function
          Voigt: fit the spectrum with a Voigt function
          PseudoV: fit the spectrum with a Pseudo-Voigt function
          Moffat: fit the spectrum with a Moffat function
        A description of all line shapes can be found at https://lmfit.github.io/lmfit-py/builtin_models.html
  spec : numpy array
        An arraycontaining the intensity axis of the spectrum.
  xaxis : numpy array
        An array containing the frequency or wavelength axis of the spectrum.
  RMS : float
        The RMS value of the data.
  showplots : bool, optional
        The indicator of whether to plot the best fit line shape or a comparison of line shapes. Default is True, which produces plots.
  plot_name : string, optional
        The name of a saved figure. Default is None, which does not save a plot.
  initial_fit_guess: string, optional.
        The path to a saved LMFIT result to use as the initial guess. Default is None; the LMFIT .guess method is used.
  weight_range : int, optional
        How many points centered around the peak frequency to give a higher weight during fit, which causes LMFIT to prioritize fitting these points. These points are given a weighting of 2x(normalized variance), all other points are given a weight of the normalized variance.
    
  Returns
  -------
  fit_result : LMFIT object
        An LMFIT ModelResult object containing the best fit results, parameters, and statistics.
  weights : numpy array
        An array containing the weights specified for each point.
  '''
  
  if weight_range is not None:
    sigma = 1/(RMS**2)
    centerpoint = np.argmax(spec) 
    weights = np.full((len(xaxis),), sigma/(len(xaxis)))
    for i in range(len(weights)):
      if i > centerpoint-weight_range and i < centerpoint + weight_range:
        weights[i] = 2*(sigma/len(xaxis))
      
  if initial_fit_guess == None:
    pars = model[fit_type]['lmfit_func'].guess(spec,x=xaxis,weights=weights)
  else:
    inital = load_modelresult(initial_fit_guess)
    pars = inital.params
  fit_result = model[fit_type]['lmfit_func'].fit(spec,pars,x=xaxis,weights = weights,scale_covar=False)
  
  return fit_result, weights

def calc_doppler_vel(peak, rest_f):
    '''
    Calculate the Doppler velocity using the Doppler Equation.
    
    Parameters
    ---------- 
    peak : float
          The calculated central peak frequency of the spectral line peak.
    rest_f : float
          The rest frequency of the transition.
    
    Returns
    -------
    v : float
          The calculated Doppler velocity in m/s.
    
    '''

    v = ((-1*peak/rest_f)+1)*c.value
    #print('The wind speed is '+str(v)+'m/s')

    return v

def errors_propagation(rest_f, std_dev):
    '''
    Calculate the wind errors through error propagation of the Doppler Equation.

    Parameters
    ----------
    rest_f : float
          The rest frequency of the transition.
    std_dev : float
          The standard deviation of the center frequency, as calculated by LMFIT.
    
    Returns
    -------
    wind_err : float
          The calculated wind error in m/s.

    '''

    a = c.value/rest_f
    wind_err = np.abs(a*std_dev)
    #print('The wind error is '+str(wind_err)+' m/s')

    return wind_err

def errors_noise_resample(iters, percentile, RMS, xaxis, fit_params, fit_data, fit_type, weights, rest_f):
    '''
    Calculate the wind errors through noise resampling methods.
    
    Parameters
    ----------
    iters : int
          The number of iterations to run for the error resampling.
    percentile : float
          The percentile at which to extract the wind error.
    RMS : float
          The RMS of the spectrum, needed for scaling of the added noise.
    xaxis : numpy array
          The x-axis used for calculating the fit.
    fit_params : LMFIT parameters object
          The parameters of the best fit, used to calculate the new models with added noise.
    fit_data : numpy array
          The best fit result of the initial LMFIT run. Noise is added to this fit for the error resampling.
    fit_type : string
          The line shape used to calculate the model.
    weights : numpy array
          The weights for each point in the spectrum.
    rest_f : float
          The rest frequency of the transition, needed for calculating the wind speed of the noisy spectra.
    
    Returns
    -------
    err_lo : float
          The lower percentile wind error.
    err_up : float
          The upper percentile wind error.
    '''
    
    MC_results = []
    for x in range(iters):
      best_fit = fit_data
      pars = fit_params
      resample = best_fit + np.random.normal(loc=0.0,scale=RMS,size=len(xaxis))
      resample_result = model[fit_type]['lmfit_func'].fit(resample,pars,x=xaxis,weights = weights,scale_covar=False)
      
      peak = resample_result.params['center']
      resample_wind = calc_doppler_vel(peak, rest_f)
      MC_results.append(resample_wind)
    
    values = sorted(MC_results)
    median = scoreatpercentile(values,50.0)
    wind_up = scoreatpercentile(values,50.0+(percentile/2))
    wind_lo = scoreatpercentile(values,50.0-(percentile/2))
    err_up = wind_up - median
    err_lo = median - wind_lo
    
    #plt.hist(values)i
    #plt.vlines(median,ymin=0,ymax=300,color='black')
    #plt.xlim(2500,3100)
    #plt.ylim(0,400)
    #plt.xlabel('Wind values (m/s)')
    #plt.title('1000 Iterations')
    #plt.show()
    return err_up, err_lo

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
  Analysis tools for a single spectrum.
    
  Example usage:
  
  from pylanetary.spectral import Spectrum
  your_spec = Spectrum(spec, freq = freq,rest_f = 345.79598990,RMS = 0.017530593)
  
  test_peak, test_std, chi, chi_red = test.fit_profile(fit_type = 'Moffat', plot_name = 'line_comparison.png')
  
  test_wind, test_err = test.calc_doppler_vel(peak = test_peak, err = test_std)
  '''
  
  ###############################################################################
  
  def __init__(self, spec, rest_f, x_space, wl = None, freq = None, RMS = None):
    '''
    Input string instantiates object using numpy arrays of the x-axis and the spectral intensity axis.
    
    Parameters
    ----------
    spec : numpy array
          The intensity/vertical axis of the spectrum
    rest_f : float
          The rest frequency of molecular transition being examined
    x_space : string
          An identifier for choosing wavelength or frequency space for the model fit. Input 'F' or 'f' for frequency space, 'W' or 'w' for wavelength space. 
    wl : numpy array, optional
          The wavelength axis of the spectrum. Default is None.
    freq: numpy array, optional
          The frequency axis of the spectrum. Default is None.
    RMS : float, optional 
          The numerical value of the data's RMS. If not specified, an RMS value is estimated through sigma clipping (see here: https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html).
    
    Attributes
    ----------
    xaxis : numpy array
          The x-axis used for all calculations. Determined by the x_space parameter.
    wl_res : float
          The wavelength resoltuion of the x-axis.
    freq_res : float
          The frequency resolution of the x-axis.
    chan_num : float
          The number of wavelength/frequency channels in the data.
    
    '''
    
    self.wl = wl
    self.freq = freq
    
    if self.wl is None and self.freq is None:
      raise ValueError('You did not provide an input for the x-axis. Please include a frequency (freq) or wavelength (wl) array.')

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
    A LMFIT wrapper to save a model fit as an initial guess.
    
    Parameters
    ----------
    fit_type : string
          The type of line shape used to generate the initial guess. A value of 'All' will result in an error.
    outfile : string
          The name of the output file save file. the file extension '.sav' will be added.
    
    '''
    
    pars = model[fit_type]['lmfit_func'].guess(self.spec,x=self.xaxis)
    fit_result = model[fit_type]['lmfit_func'].fit(self.spec,pars,x=self.xaxis,scale_covar=False)
    #print(fit_result.fit_report())
    save_modelresult(fit_result, outfile+'.sav')
    
  ###############################################################################
    
  def fit_profile(self, fit_type, print_results = False, showplots = True, plot_name = None, initial_fit_guess = None, weight_range = None):
    '''
    Calculate the best fit of a Spectral object using the LMFIT algorithm.
    
    Parameters
    ----------
    fit_type : string
          The type of line shape used to fit the spectra, string. Options are:
            All: Fit the spectrum with all line 5 line shape options. Data is always plotted.
            Gaussian: fit the spectrum with a Gaussian function
            Lorentzian: fit the spectrum with a Lorentzian function
            Voigt: fit the spectrum with a Voigt function
            PseudoV: fit the spectrum with a Pseudo-Voigt function
            Moffat: fit the spectrum with a Moffat function
            A description of all line shapes can be found at https://lmfit.github.io/lmfit-py/builtin_models.html
    showplots : bool, optional
          The indicator of whether to plot the best fit line shape or a comparison of line shapes. Default is True, which produces plots.
    plot_name : string, optional
          The name of a saved figure. Default is None, which does not save a plot.
    initial_fit_guess: string, optional.
          The path to a saved LMFIT result to use as the initial guess. Default is None; the LMFIT .guess method is used.
    weight_range : int, optional
          How many points centered around the peak frequency to give a higher weight during fit, which causes LMFIT to prioritize fitting these points. These points are given a weighting of 2x(normalized variance), all other points are given a weight of the normalized variance.
    
    Returns
    -------
    peak : float
          The frequency or wavelength value of the line peak, which is LMFIT's 'center' parameter.
    err : float
          The standard deviation of the peak frequency or wavelength value.
    chi :
          The calculated Chi-squared statistic of the line fit.
    chi_red :
          The calculated reduced Chi-squared statistic of the line fit.
    
    Notes
    -----
    Guide to keys in the nested model dictionary:
          model[] : 
                The Model name, taken from the user input.
          ref : 
                A short reference name of model type.
          lmfit_func: 
                The built-in LMFIT function name, needed to properly interface with the LMFIT package.
          label :
                The label for the legend included in the final plots.
          color : 
                The color of the line for a given fit type when plotting.
    
    An initial fit guess is not necessary, but can be very beneficial if the spectra are noisy. If your fit is failing, try including an intial guess to improve the performance of LMFIT. It is recommended that you use a pixel with as small a Doppler shift as possible to make your initial guess save file.
    
    '''
    
    if fit_type != 'All':
      if initial_fit_guess == None:
        pars = model[fit_type]['lmfit_func'].guess(self.spec,x=self.xaxis)
      else:
        inital = load_modelresult(initial_fit_guess)
        pars = inital.params
      fit_result, weights = lmfit_wrapper(fit_type, self.spec, self.xaxis, self.RMS, initial_fit_guess, weight_range)
      
      if print_results is True:
        print(fit_result.fit_report())
        
    elif fit_type == 'All':
      fit_result = {}
      for key in model:
        pars = model[key]['lmfit_func'].guess(self.spec,x=self.xaxis)
        result, weights = lmfit_wrapper(key, self.spec, self.xaxis, self.RMS, initial_fit_guess, weight_range)
        #result = model[key]['lmfit_func'].fit(self.spec,pars,weights = weights,x=self.xaxis,scale_covar=False)
        print(result.fit_report())
        fit_result[key] = result
    else:
      raise ValueError('fit_type must be one of the following options: Gaussian, Lorentzian, Voigt, PseudoV, Moffat, or All')
    
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
      best_fit_params = fit_result.params
      best_fit_data = fit_result.best_fit
    else:
      peak = 0
      err = 0
      best_fit = np.NAN
      chi = np.NAN
      chi_red = np.NAN
      best_fit_params = np.NAN
      best_fit_data = np.NAN
      weights = np.NAN
    
    return peak, err, chi, chi_red, best_fit_params, best_fit_data, weights
    
  ############################################################################### 
  
  def plot_lines(self, line_result, model_key, x_label = 'Frequency (GHz)', y_label = 'Intensity (mJy)', residual_label = 'Residuals', plot_title = 'Comparison of Line Shapes'):
    '''
    Plot the original data and best fit line shape.
    
    Parameters
    ----------
    line_result : LMFIT fit object
          The LMFIT best fit result, calculated using fit_profile.
    model_key : string
          The name of the line shape used to calculate the fit.
    x_label : string, optional
          The x-axis label for the main and residual plot. Default is 'Frequency (GHz)'.
    y_label : string, optional
          The y-axis label for the main plot. Default is 'Intensity (mJy)'.
    residual_label : string, optional.
          The y-axis label for the residual plot. Default is 'Residuals'.
    plot_title : string, optional
          The title for the plot. Default is 'Comparison of Line shapes'.
    
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
  
# This section is used for testing purposes during development. It will be removed when the code is ready to be fully merged with the main branch.
   
# Simple single spectra testing case
#if __name__=="__main__":
  
  # CO [10,25]
  #t0 = time.time()
  #freq = np.array([345.76770566,345.76868215,345.76965863,345.77063511,345.77161159,345.77258807,345.77356455,345.77454103,345.77551751,345.77649399,345.77747047,345.77844695,345.77942343,345.78039991,345.78137639,345.78235287,345.78332935,345.78430584,345.78528232,345.7862588,345.78723528,345.78821176,345.78918824,345.79016472,345.7911412,345.79211768,345.79309416,345.79407064,345.79504712,345.7960236,345.79700008,345.79797656,345.79895304,345.79992953,345.80090601,345.80188249,345.80285897,345.80383545,345.80481193,345.80578841,345.80676489,345.80774137,345.80871785,345.80969433,345.81067081,345.81164729,345.81262377,345.81360025,345.81457673,345.81555322,345.8165297,345.81750618,345.81848266,345.81945914,345.82043562,345.8214121,345.82238858])
  #spec = np.array([0.21198112,0.18924561,0.21650964,0.22294408,0.24960007,0.22984277,0.25219446,0.24985315,0.25343025,0.28705364,0.3061584,0.31352046,0.3277193,0.3265509,0.3436137,0.35133913,0.38450843,0.4165357,0.43355167,0.467518,0.49524495,0.5277506,0.5910206,0.64392966,0.7271694,0.89640355,0.8984462,0.76777405,0.6531943,0.59134954,0.55332845,0.50916755,0.46389887,0.4410508,0.4036594,0.40885097,0.37320843,0.3508784,0.34755984,0.31414196,0.29820374,0.27975425,0.28136522,0.26828533,0.26605716,0.24770543,0.2537956,0.22927019,0.219696,0.21160068,0.20004132,0.1939475,0.18222019,0.16944197,0.17536835,0.16506103,0.16306329])
  
  #test = Spectrum(spec, freq = freq,rest_f = 345.79598990,x_space = 'f', RMS = 0.017530593)
  
  #test.make_initial_guess(fit_type = 'Moffat', outfile = 'moffat_guess')
  #test_peak, test_std, chi, chi_red, test_best_params, test_best_data, weightings = test.fit_profile(fit_type = 'Voigt', weight_range=7, showplots = False)
  # initial_fit_guess = 'moffat_guess.sav'
  #test_wind = calc_doppler_vel(peak = test_peak, rest_f = 345.79598990)
  #test_err = errors_propagation(rest_f,test_std)
  #test_err_up, test_err_lo = errors_noise_resample(300, 68, test.RMS, test.xaxis, test_best_params, test_best_data, 'Moffat', weightings, 345.79598990)
  #print(test_err_up, test_err_lo)
  #tf = time.time()
  #print('Time to run single specturm: ')
  #print(tf-t0)
###############################################################################
###############################################################################
  
class SpectralCube:
  '''
  Analysis tools for a FITS image cube.
  
  Example Usage, full image analysis:
  
  cube = SpectralCube(your_image_path, x_space = 'f')
  
  data = cube.extract_image()
  cube.fit_data(data, fit_type = 'Moffat', outfile = 'development_testing_fit', RMS = 0.017530593, SN = 6,initial_fit_guess = '/homes/metogra/skathryn/Research/Scripts/co_moffat_modelresult.sav')
  test_cube.wind_calc(picklefile = 'development_testing_fit.pickle', restfreq = 345.79598990, outfile = 'development_testing_wind')
  test_cube.plot_data(picklefile = 'development_testing_wind.pickle', platescale = 0.1, body = 'Neptune', title = 'CO Doppler Velocity Map', cont_label = 'Radial Velocity (m/s)', date = '2016-04-30 00:00', location = 'ALMA', spatial = 4e4, limits = [-2000,2000], contours = [21,200])
  
  '''
  ###############################################################################
  
  def __init__(self, imagepath, x_space):
    '''
    Input string instantiates object using the FITS image path.
    
    Parameters
    ----------
    imagepath : string
          The path to FITS image file.
    x_space : string
          The identifier for choosing wavelength or frequency space for the model fit. Input 'F' or 'f' for frequency space, 'W' or 'w' for wavelength space.
    
    Attributes
    ----------
    chan_num : float
          The number of frequency channels. Pulled from image header.
    i0 : float
          The reference pixel of the image. Pulled from image header.
    df : float
          The channel width in Hz. Pulled from image header.
    f0 : float
          The rest frequency of the image in Hz. Pulled from image header.
    xpixmax : float
          The x-dimension size in pixels. Pulled from image header.
    ypixmax : float
          The y-dimension size in pixels. Pulled from image header.
    bmaj : float
          The semi-major axis of the restoring beam in arcsec. Pulled from image header. 
    bmin : float
          The semi-minor axis of the restoring beam in arcsec. Pulled from the image header.
    bpa : float
          The position angle of the restoring beam in degrees. Pulled from the image header.
    data : numpy array
          The spectral data pulled from the image cube.
    freqspec : numpy array
          The frequency grid around the rest frequency, which can be used as the x-axis.
    wl_spec : numpy array
          The wavelength grid around the rest wavelength, which can be used as the x-axis.
    
    '''  
    self.x_space = x_space
    
    fits_file = fits.open(imagepath)
    hdr = fits_file[0].header
    
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
      
    if self.x_space == 'F' or self.x_space == 'f':
      self.xaxis = self.freqspec
    elif self.x_space == 'W' or self.x_space == 'w':
      self.xaxis = self.wl_spec
    else:
      raise ValueError('You did not choose frequency or wavelength space. Please input F/f for frequency, W/w for wavelength.')
      
  ###############################################################################
  
  def extract_pixel(self, pixel):
    '''
    Extract the spectral axis for a single pixel.
    
    Parameters
    ----------
    pixel : list
          The indexes of the pixel to be fitted.
    
    Returns
    -------
    datas : numpy array
          The spectral axis of the pixel.
    
    '''
    print('Fitting pixel ['+str(pixel[0])+','+str(pixel[1])+']')
    datas = np.empty((1,1),dtype=np.ndarray)
    
    print('Extracting spectrum at pixel '+str(pixel[0])+','+str(pixel[1]))
    datas = self.data[0,:,pixel[1],pixel[0]]
    
    return datas
  
  ###############################################################################
    
  def extract_image(self):
    '''
    Extract the spectral axis for the entire image cube.
    
    Returns
    -------
    datas : numpy array
          The 3D array containing the spectral axis for each pixel.
    
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
    Make a boolean array mask from extracting a sub-image of pixels.
    
    Parameters
    ----------
    pixels : tuple
          The pixel values to include in the mask. These pixels will be given a value of True and their spectra extracted.
    
    Returns
    -------
    mask : bool
          A 2D boolean array where pixels with values of True are extracted, and values of False are not.
    
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
    Extract pixels defined in a boolean array.
    
    Parameters
    ----------
    mask : bool
          2D boolean array of the size [xpixmax,ypixmax]. Pixels with a value of True are extracted for fitting.
    
    Returns
    -------
    datas : numpy array
          The 3D numpy array containing the spectral axis. Masked pixels are NAN. 
    
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
  
  def fit_data(self, datas, fit_type, outfile, initial_fit_guess = None, RMS = None, SN = None, weight_range = None):
    '''
    Calculate the best fit for the spectra at each pixel.
    
    Parameters
    ----------
    datas : numpy array
          The array of spectral data to be fit.
    fit_type : string
          The type of line shape used to fit the spectra. Options are Gaussian, Lorentzian, Voigt, PseudoV, Moffat, and All. It is not recommended that you use the option of all for anything more than a single pixel.
    outfile : string
          The name of the saved output pickle file. The 'pickle' file extension is automatically included.
    initial_fit_guess : string, optional
          The name of a saved LMFIT result to use as an intitial guess instead of the LMFIT guess function. Default is None.
    RMS : float, optional
          The numerical value of the data's RMS. Default is None; if not provided, an estimate is calculated using Astropy's siga_clipped_stats (https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html)
    SN : float
          The Signal to noise ratio minimum, which is used to mask out noisy data. Any pixel with a S/N ratio lower than this calue will not be fit. Default is value is 1. 
    errors : string, optional
          The method for calculating the errors in the wind speed. Options include error propagation (prop), noise resampling (resample), covariances (covar), and MCMC (MC). The default is error propagation.
    
    Returns
    -------
    weights : numpy array
          An array containing the weightings of each spectral point used in the fit. This is necessary for the error resampling function.
    '''
    
    if SN == None:
      SN = 1
    
    picklefile = open(outfile+'.pickle', 'wb')
    print('Numerical results will be saved in the '+outfile+'.pickle file')
    results = {'x':[],'y':[],'center':[],'std':[],'chi':[],'redchi':[],'fit_params':[],'fit_data':[]}
    
    if datas.ndim == 1:
      spectra = Spectrum(spec = datas, freq = self.xaxis, rest_f = 345.79598990, x_space = self.x_space, RMS = RMS) 
      peak, std, chi, chi_red, best_fit_params, best_fit_data, weights = spectra.fit_profile(fit_type = fit_type, print_results = True, initial_fit_guess = initial_fit_guess, showplots = False, weight_range = weight_range)
      
      fit_params = best_fit_params.dumps()
      
      results['center'].append(peak)
      results['std'].append(std) 
      results['chi'].append(chi)
      results['redchi'].append(chi_red)
      results['fit_params'].append([fit_params])
      results['fit_data'].append([best_fit_data])
      
    else:
    
      for xpix in range (0,self.xpixmax,1):
        for ypix in range (0,self.ypixmax,1): 
          s_n = abs(np.max(datas[xpix,ypix])/RMS)
          if s_n > SN:
            print('Modeling spectrum at pixel '+str(xpix)+','+str(ypix))
            spectra = Spectrum(spec = datas[xpix,ypix], freq = self.xaxis, rest_f = 345.79598990, x_space = self.x_space, RMS = RMS) 
            peak, std, chi, chi_red, best_fit_params, best_fit_data, weights = spectra.fit_profile(fit_type = fit_type, print_results = True, initial_fit_guess = initial_fit_guess, showplots = False, weight_range = weight_range)
            fit_params = best_fit_params.dumps()
          
            results['x'].append(xpix)
            results['y'].append(ypix)
            results['center'].append(peak)
            results['std'].append(std) 
            results['chi'].append(chi)
            results['redchi'].append(chi_red)
            results['fit_params'].append([fit_params])
            results['fit_data'].append([best_fit_data])
          
          elif s_n <= SN or np.isnan(s_n): 
            results['x'].append(xpix)
            results['y'].append(ypix)
            results['center'].append(np.NAN)
            results['std'].append(np.NAN) 
            results['chi'].append(np.NAN)
            results['redchi'].append(np.NAN)
            results['fit_params'].append(np.NAN)
            results['fit_data'].append(np.NAN)
      results['fit_params'] = np.array(results['fit_params'], dtype = object)
      results['fit_data'] = np.array(results['fit_data'],dtype=object)
          
      lenx = np.shape(datas)[0]
      leny = np.shape(datas)[1]
    
      results['x']=np.reshape(results['x'],(lenx,leny))
      results['y']=np.reshape(results['y'],(lenx,leny))
      results['center']=np.reshape(results['center'],(lenx,leny))
      results['std']=np.reshape(results['std'],(lenx,leny))
      results['chi']=np.reshape(results['chi'],(lenx,leny))
      results['redchi']=np.reshape(results['redchi'],(lenx,leny))
      results['fit_params']=np.reshape(results['fit_params'],(lenx,leny))
      results['fit_data']=np.reshape(results['fit_data'],(lenx,leny))
      
    pickle.dump(results,picklefile)
    picklefile.close()
    
    return weights
    
  ###############################################################################  
  
  def wind_calc(self, picklefile, restfreq, outfile):
    '''
    Calculate the Doppler wind for each pixel.
    
    Parameters
    ----------
    picklefile : string
          The path to the pickle file of calculated doppler shifts.
    restfreq : float
          The rest frequency of the molecular transition.
    outfile : string
          The name of the wind results' pickle file.
    
    '''
    
    r = pickle.load(open(picklefile,"rb"),encoding='latin-1') 
    peaks = np.array(r['center'])
    std = r['std']
    
    x_max = peaks.shape[0]
    y_max = peaks.shape[1]
    
    new_pickle = open(outfile+'.pickle', 'wb')
    print('Numerical wind results will be saved in the '+outfile+'.pickle file')
    results = {'x':[],'y':[],'v':[]}
    
    for xpix in range (0,x_max,1):
      for ypix in range (0,y_max,1):
        v = calc_doppler_vel(peaks[xpix,ypix], restfreq)
        
        results['x'].append(xpix)
        results['y'].append(ypix)
        results['v'].append(v)
    
    lenx = np.shape(r['center'])[0]
    leny = np.shape(r['center'])[1]
    
    results['x']=np.reshape(results['x'],(lenx,leny))
    results['y']=np.reshape(results['y'],(lenx,leny))  
    results['v']=np.reshape(results['v'],(lenx,leny))
    
    pickle.dump(results,new_pickle)
    new_pickle.close()
        
  ###############################################################################
  
  def calc_wind_error(self, calc_type, picklefile, iters = None, percentile = None, RMS = None, xaxis = None, fit_type = None, weights = None, rest_f = None):
    '''
    Calculate the errors in the Doppler wind speed.
    
    Parameters
    ----------
    calc_type : string
          The type of error calculation used. Options include error propagation (prop), noise resampling (resample), covariances (covar), and MCMC (MC).
    picklefile : string
          The path to an existing pickle file where the errors will be saved. For the propagation method, this file should contain the standard deviation of the center frequency to calculate the errors.
    iters : int, optional
          The number of iterations to run for the error resampling.
    percentile : float, optional
          The percentile at which to extract the wind error. Needed for error resampling
    RMS : float, optional
          The RMS of the spectrum, needed for scaling of the added noise when conducting error resampling.
    xaxis : numpy array, optional
          The x-axis used for calculating the fit.
    fit_type : string, optional
          The line shape used to calculate the model.
    weights : numpy array, optional
          The weights for each point in the spectrum.
    rest_f : float, optional
          The rest frequency of the transition, needed for calculating the wind speed of the noisy spectra.
          
    Returns
    -------
    
    
    Notes
    -----
    Parameters needed for the error propagation method:
    - calc_type
    - picklefile
    
    Parameters needed for the noise resampling method:
    - calc_type
    - iters
    - percentile
    - RMS
    - xaxis
    - fit_params
    - fit_data
    - fit_type
    - weights
    - rest_f
    
    Parameters needed for the covariance method:
    
    Parameters needed for the MCMC method:
    
    
    '''
    results = pickle.load(open(picklefile,"rb"),encoding='latin-1')
    
    peak_std = results['std']
    fit_params = results['fit_params']
    fit_data = results['fit_data']
    
    x_max = fit_params.shape[0]
    y_max = fit_params.shape[1]
    
    results['v_err_up'] = []
    results['v_err_lo'] = []
    for xpix in range (0,x_max,1):
      for ypix in range (0,y_max,1):
        if calc_type == 'prop':
          v_err_up = v_err_lo = errors_propagation(rest_f,peak_std[xpix,ypix])      
        elif calc_type == 'resample':
            if type(fit_params[xpix,ypix]) == float:
              v_err_up = v_err_lo = np.NAN
            elif type(fit_params[xpix,ypix]) == list:
              print('Calculating errors for pixel '+str(xpix)+','+str(ypix))
              fit_param = Parameters()  
              s = fit_params[xpix,ypix][0]
              fit_param = fit_param.loads(s)
              v_err_up, v_err_lo = errors_noise_resample(iters, percentile, RMS, xaxis, fit_param, fit_data[xpix,ypix], fit_type, weights, rest_f)
        elif calc_type == 'covar':
          continue
        elif calc_type == 'MC':
          continue
        else:
          raise ValueError('You did not choose an available error calculation method. Please try again')
        
        results['v_err_up'].append(v_err_up)
        results['v_err_lo'].append(v_err_lo)
        
    lenx = np.shape(results['x'])[0]
    leny = np.shape(results['x'])[1]
    results['v_err_up']=np.reshape(results['v_err_up'],(lenx,leny))    
    results['v_err_lo']=np.reshape(results['v_err_lo'],(lenx,leny))

    new_pickle = open(picklefile, 'wb')
    pickle.dump(results,new_pickle)
    new_pickle.close()
    
    
  ###############################################################################  
  
  def plot_data(self, picklefile, platescale, body, cont_label, title = None, date = None, location = None, spatial = None, variable = None, cmap = None, limits = None, contours = None, hatch = None, savefig = None):
    '''
    Plot the calculated winds on a 2D plane. 
    This algorithm was originally written by Dr. Martin Cordiner for use on Titan (Cordiner et al. 2020).
    
    Parameters
    ----------
    picklefile : string
          The path to the pickle file of calculated doppler winds and errors.
    platescale : float
          The platescale of the data.
    body : string
          The name of the planetary object you are studying.
    cont_label : string
          The label for the contour bar, which corresponds to the pixel color.
    title : string, optional
          The plot title. Default is None.
    date : string, optional
          The date of the observation in 'YYYY-MM-DD 00:00' format. Default is the current date.
    location : string, optional
          The name of the observing device, the JPL observatory code. Default is the center of the Earth.
    spatial : float, optional
          The numerical value for spatial axes bounds in km. If not specified, the value is selected so the image is 1.5 planet diameters wide.
    variable : string, optional
          The name of the variable, either winds or errors, to be plotted. Default is the Doppler velocities (v). Options include 'v', 'v_err_up', and 'v_err_lo'.
    cmap : string, optional
          The name of Matplotlib colormap for plotting. Default is blue/white/red gradient. 
    limits : list, optional
          The limits of the wind colorbar, list. Default is 0 to 1.
    contours : list, optional
          The defintion of the contour lines. The first value is the number of lines to plot. Second value is the multiplicative factor for the contor values. Example: for [13,200] specifices 13 contour lines between -2200 and 2200 m/s. Default is [11,10] (11 contour lines from -100 to 100)
    hatch : bool, optional
          An indicator of if the restoring beam is hatched or not. Default is False (no hatch marks).
    savefig : string, optional
          The file name/path for the saved figure. Default is None, which does not save a figure.
    
    '''
    
    planet = Body(body)
    dist = planet.semi_major_axis.value
    planetrad = planet.req.value
    subobslat = planet.ephem['PDObsLat']
    ccw = planet.ephem['NPole_ang']
    
    # Define basic plotting parameters
    matplotlib.rcParams["contour.negative_linestyle"]='solid'
    planetgridcolor='black'
    kmscale=725.27*dist*platescale
    
    if spatial is None:
      spatial = planetrad*1.25
    
    if cmap == None: 
      cmap = LinearSegmentedColormap.from_list('mycmap', (['blue','white','red']))
    else:
      cmap = cmap
    
    if limits == None:
      norm = None
    else:
      norm = colors.Normalize(vmin=limits[0],vmax=limits[1])
      
    if contours == None:
      levels = np.array(centered_list(10)) * 10
    else:
      levels = np.array(centered_list(contours[0])) * contours[1]
    
    # Load the data
    r = pickle.load(open(picklefile,"rb"),encoding='latin-1') 
    if variable == None:
      data = r['v']
    else:
      data = r[variable]
    
    subobslat = np.deg2rad(subobslat)
    ccw = np.deg2rad(ccw)
    
    
    x = ((r['x']-len(r['x'])/2.))*kmscale
    y = (r['y']-len(r['y'])/2.)*kmscale
    
    # Initialize the figure
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    c=ax.imshow(data.transpose(),extent=[x.min(), x.max(), y.min(), y.max()],origin='lower',interpolation='nearest',norm=norm,zorder=-21,cmap=cmap)
    co=ax.contour(data.transpose(),levels=levels,extent=[x.min(), x.max(), y.min(), y.max()],origin='lower',colors='k',linewidths=0.75,zorder=-20)
    
    ax.clabel(co,fmt="%d",fontsize=13)
    ax.set_xlabel('Distance (km)', fontsize = 20)
    ax.set_ylabel('Distance (km)', fontsize = 20)
    ax.tick_params(axis='both',labelsize=18)
    ax.set_title(title, fontsize = 24)
    cb = fig.colorbar(c)
    cb.set_label(cont_label, fontsize = 20)
    cb.ax.tick_params(labelsize = 20)
    
    # Add the beam ellipse
    width = self.bmin*725.27*dist
    height = self.bmaj*725.27*dist
    beam = Ellipse(xy=[-3.0e4,-3.2e4], width=self.bmin*725.27*dist, height=self.bmaj*725.27*dist, angle=self.bpa)
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
        dprod = (northx*x + northy*y + northz*zcoord)/(planetrad)
        z_lat = 90. - np.degrees(np.arccos(dprod))
    ax.contour(z_lat,colors=planetgridcolor,extent=[-spatial, spatial, -spatial, spatial],linestyles='dashed',zorder=0,alpha=0.5,levels=latlevel)
    ax.contour(z_lat,colors=planetgridcolor,extent=[-spatial, spatial, -spatial, spatial],levels=[0],linestyles='solid',zorder=0)

    #Longitude lines
    xma=np.ma.masked_where(np.isnan(zcoord),x)
    yma=np.ma.masked_where(np.isnan(zcoord),y)
    
    xr = np.cos(ccw)*xma + np.sin(ccw)*yma
    yr = -np.sin(ccw)*xma + np.cos(ccw)*yma
    
    projz = (zcoord - (yr * np.tan(subobslat))) * np.cos(subobslat)
    z_long = np.rad2deg(np.arctan2(xr,projz))
    
    projx = (xma - (-yma * np.tan(ccw))) * np.cos(ccw) 
    projz = (zcoord - (yma * np.tan(subobslat))) * np.cos(subobslat)
    ax.contour(z_long,12,colors=planetgridcolor,extent=[-spatial, spatial, -spatial, spatial],linestyles='dotted',zorder=0)
    
    plt.show()
    if savefig == None:
      print('Figure was not saved')
    else:
      fig.savefig(savefig.png, bbox_inches='tight')
      
  ###############################################################################

# This section is used for testing purposes during development. It will be removed when the code is ready to be fully merged with the main branch.
if __name__=="__main__":
  
  image = '/homes/metogra/skathryn/Research/Data/ContSub/CO/CS20/Neptune_Pri_X50a4_CS20_narrow_square_2.fits'
  test_cube = SpectralCube(image,x_space = 'f')
  
  # full image analysis
  
  test_data = test_cube.extract_image()
  #test_data = test_cube.extract_pixel([16,16])
  weights = test_cube.fit_data(test_data, fit_type = 'Moffat', outfile = 'development_testing_fit', RMS = 0.017530593, SN = 6,weight_range = 7,initial_fit_guess = '/homes/metogra/skathryn/Research/Scripts/co_moffat_modelresult.sav')
  test_cube.wind_calc(picklefile = 'development_testing_fit.pickle', restfreq = 345.79598990, outfile = 'development_testing_wind')
  t0 = time.time()
  #test_cube.calc_wind_error(calc_type = 'prop', picklefile = 'development_testing_fit.pickle',rest_f = 345.79598990)
  test_cube.calc_wind_error(calc_type = 'resample', picklefile = 'development_testing_fit.pickle', iters = 1000, percentile = 68, RMS = 0.017530593, xaxis = test_cube.xaxis, fit_type = 'Moffat', weights = weights, rest_f = 345.79598990)
  t1 = time.time()
  test_cube.plot_data(picklefile = 'development_testing_fit.pickle', platescale = 0.1, body = 'Neptune', title = 'CO Doppler Velocity Error Map - Error Prop', cont_label = 'Radial Velocity Error (m/s)', date = '2016-04-30 00:00', location = 'ALMA', spatial = 4e4, limits = [0,200], variable = 'v_err_up', cmap = 'inferno_r')
  
  # sub-image analysis - 3x3 sub-image at the center
  
  #mask_pix = [[15,17],[16,17],[17,17],[15,16],[16,16],[17,16],[15,15],[16,15],[17,15]]
  #mask = test_cube.make_mask(mask_pix)
  #data = test_cube.extract_mask_region(mask)
  
  #test_cube.fit_data(data, fit_type = 'Moffat', outfile = 'development_testing_fit_subim', RMS = 0.017530593, SN = 6,initial_fit_guess = '/homes/metogra/skathryn/Research/Scripts/co_moffat_modelresult.sav')
  #test_cube.wind_calc(picklefile = 'development_testing_fit_subim.pickle', restfreq = 345.79598990, outfile = 'development_testing_wind_subim')
  #test_cube.plot_data(picklefile = 'development_testing_wind_subim.pickle', platescale = 0.1, body = 'Neptune', title = 'CO Doppler Velocity Map Sub-Image', cont_label = 'Radial Velocity (m/s)', date = '2016-04-30 00:00', location = 'ALMA', spatial = 4e4, limits = [-2000,2000])
  
  print('Time to run resample: ')
  print(t1-t0)
  
