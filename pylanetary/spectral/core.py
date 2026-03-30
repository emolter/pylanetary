import os
import numpy as np
from dataclasses import dataclass
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel, MoffatModel
from lmfit.model import save_modelresult, load_modelresult
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.constants import c
from astropy.time import Time
from pylanetary.utils import *
from scipy.signal import savgol_filter
import imageio.v2 as imageio
from matplotlib import pyplot as plt
import pickle


# =========================
# Dataclasses for Spectrum()
# =========================
@dataclass
class SpectrumData:
    """
    Container for a single spectrum input data.
    """
    intensity: np.ndarray
    frequency: np.ndarray
    nu_rest: float = None
    i: int = None
    j: int = None
    weights: np.ndarray = None

@dataclass
class FitResult:
    """
    Container for fitted spectral line results.
    """
    amplitude: float = None
    center: float = None
    sigma: float = None
    beta: float = None


# =========================
# Model dictionary
# =========================
model = {'Gaussian': {'ref': 'gauss', 'lmfit_func': GaussianModel(), 'label': 'Gaussian Fit', 'color': 'orange'},
         'Lorentzian': {'ref': 'lor', 'lmfit_func': LorentzianModel(), 'label': 'Lorentzian Fit', 'color': 'green'},
         'Voigt': {'ref': 'voigt', 'lmfit_func': VoigtModel(), 'label': 'Voigt Fit', 'color': 'red'},
         'PseudoV': {'ref': 'psv', 'lmfit_func': PseudoVoigtModel(), 'label': 'Pseudo-Voigt Fit', 'color': 'purple'},
         'Moffat': {'ref': 'mof', 'lmfit_func': MoffatModel(), 'label': 'Moffat Fit', 'color': 'brown'}
         }

# TODO This could be a separate constants.py file that is imported and modified outside of the module or a config file.
# =========================
# Rest Frequency Constants
# =========================
REST_FREQUENCIES = {
    'CO10': 115.271201796,  # in GHz
    'CO21': 230.538,        # in GHz
    'CO32': 345.795989873,  # in GHz or 345.7959899 in Splatologue
    'HCN32': 265.8864343,   # in GHz
    'HCN43': 354.5054779,   # in GHz
    'HC3N': 363.785397      # in GHz J=40-39 363.785397 in Splatologue
    }


#########################
# The start of Spectrum()
#########################
class Spectrum:
    """
    Class for a single spectrum from a spectral cube. Spectrum has tools to best fit a single
    molecular transition in the spectral axis called nu_arr in GHz with profiles of varying
    shapes in spectral intensity called I_arr.
    """

    def __init__(self, data: SpectrumData, snr_threshold: float = 3.0):

        self.data = data

        self.I_nu_arr = data.intensity
        self.nu_arr = data.frequency
        self.nu_rest = data.nu_rest

        self.i = data.i
        self.j = data.j

        self.weights = data.weights
        self.snr = snr_threshold

        self.del_nu = self.nu_arr[1] - self.nu_arr[0]
        self.num_chan = np.size(self.nu_arr)

        self.mu, self.median, self.std = sigma_clipped_stats(self.I_nu_arr)
        self.rms = np.sqrt(np.mean(np.square(self.I_nu_arr - self.mu)))

        self.nu_at_Imax = None
        self.Imax_value = None

        self.signal_flag = False
        self.filtered_snr_data = None
        self.filtered_snr_data_max_value = None
        self.filtered_snr_data_window_length = None

        self.x_to_fit = None
        self.y_to_fit = None
        self.y_for_initial_guess = None

        self.lmfit_initial_guess_model = None
        self.lmfit_fit_result_model = None

        self.fit_result = FitResult()

        self.nu_low_bound = None
        self.nu_high_bound = None

    def fit_profile(self,
                    fit_type,
                    guess_smooth_frac,
                    bound_fit,
                    bound_velocity_maximum,
                    use_weights,
                    weight_sigma,
                    plot_pixel_fit,
                    plot_dir='./plots'):
        """
        Calculate the best fit of a Spectrum object using the LMFIT package.
        """

        self.fit_result = FitResult()

        # --- Standard smoothing for SNR detection ---
        # TODO this could be more complicated focusing on a window around the expected line center
        self.filtered_snr_data, self.filtered_snr_data_window_length = smooth_data(
            data_to_smooth=self.I_nu_arr
        )

        self._check_signal_rms(input_data=self.filtered_snr_data)

        if self.signal_flag:

            if np.any(np.isnan(self.I_nu_arr)) or np.any(np.isnan(self.nu_arr)):
                print("Intensity or frequency array contains NaN values. Aborting fit.")
                return None

            # --- Define fitting domain ---
            if bound_fit:
                self._apply_bounds(bound_velocity_maximum)
                self.y_for_initial_guess, _ = smooth_data(
                    data_to_smooth=self.y_to_fit,
                    fraction=guess_smooth_frac,
                    polyorder=2
                )
            else:
                self.x_to_fit = self.nu_arr.copy()
                self.y_to_fit = self.I_nu_arr.copy()
                self.y_for_initial_guess, _ = smooth_data(
                    data_to_smooth=self.I_nu_arr,
                    fraction=guess_smooth_frac,
                    polyorder=2
                )

            # --- Initial guess ---
            self._make_initial_guess(
                input_data=self.y_for_initial_guess,
                input_x=self.x_to_fit,
                fit_type=fit_type
            )

            pars = self.lmfit_initial_guess_model.params

            # --- Weights ---
            if use_weights:
                self._make_weights(weight_sigma)

            # --- Fit ---
            try:
                self.lmfit_fit_result_model = model[fit_type]['lmfit_func'].fit(
                    self.y_to_fit,
                    pars,
                    x=self.x_to_fit,
                    weights=self.weights,
                    scale_covar=False,
                    nan_policy='omit'
                )
            except ValueError as e:
                print("No suitable fit found:", e)
                return None

            # --- Extract results ---
            if self.lmfit_fit_result_model is not None:
                self.fit_result = FitResult(
                    amplitude=self.lmfit_fit_result_model.best_values.get('amplitude'),
                    center=self.lmfit_fit_result_model.best_values.get('center'),
                    sigma=self.lmfit_fit_result_model.best_values.get('sigma'),
                    beta=self.lmfit_fit_result_model.best_values.get('beta')
                )

            if self.y_to_fit is not None and len(self.y_to_fit) > 0:
                max_index = np.argmax(self.y_to_fit)
                self.nu_at_Imax = self.x_to_fit[max_index]
                self.Imax_value = self.y_to_fit[max_index]

                if plot_pixel_fit and self.signal_flag:
                    generate_diagnostic_plot(nu_arr=self.nu_arr,
                                                I_nu_arr=self.I_nu_arr,
                                                filtered_snr_data=self.filtered_snr_data,
                                                filtered_snr_data_max_value=self.filtered_snr_data_max_value,
                                                filtered_snr_data_window_length=self.filtered_snr_data_window_length,
                                                x_to_fit=self.x_to_fit,
                                                y_to_fit=self.y_to_fit,
                                                y_for_initial_guess=self.y_for_initial_guess,
                                                lmfit_initial_guess_model=self.lmfit_initial_guess_model,
                                                lmfit_fit_result_model=self.lmfit_fit_result_model,
                                                nu_low_bound=self.nu_low_bound,
                                                nu_high_bound=self.nu_high_bound,
                                                weights=self.weights,
                                                nu_at_Imax=self.nu_at_Imax,
                                                Imax_value=self.Imax_value,
                                                snr=self.snr,
                                                rms=self.rms,
                                                i=self.i,
                                                j=self.j,
                                                plot_dir=plot_dir
                                            )

        return self.fit_result


    def _check_signal_rms(self, input_data):
        """
        Check the input data for a spectral signal intensity above the pre-determined SNR threshold.
        """
        self.filtered_snr_data_max_value = np.max(input_data)
        threshold = self.rms * self.snr
        if self.filtered_snr_data_max_value > threshold:
            self.signal_flag = True

    def _apply_bounds(self, Doppler_maximum_velocity=7e4):
        """
        Method to return frequency bounds to reduce domain of fit for some realistic limit to the maximum
        wind speed possible.

        Parameters
        ----------
        Doppler_maximum_velocity: float; default = 7e4 m/s
            The maximum physical Doppler velocity in m/s to limit the spectrum profile fitting in frequency.
        """
        self.nu_high_bound = self.nu_rest * (Doppler_maximum_velocity / c.value + 1)
        self.nu_low_bound = -self.nu_rest * (Doppler_maximum_velocity / c.value - 1)

        spectral_fit_mask = (self.nu_arr >= self.nu_low_bound) & (self.nu_arr <= self.nu_high_bound)
        self.x_to_fit = self.nu_arr[spectral_fit_mask]
        self.y_to_fit = self.I_nu_arr[spectral_fit_mask]

    def _make_weights(self, weight_sigma=None):
        """
        Calculate and store Gaussian weights to fit the spectral profile.

        Parameters
        ----------
        weight_sigma : float, optional
            The spread (standard deviation in GHz) of the Gaussian used to calculate weights.
            The spread defaults to ten resolution elements around the center rest frequency.

        Notes
        -----
        The weights are calculated as a Gaussian function of the spectral frequency array
        (self.x_to_fit). Higher weights are assigned to data points closer to the rest frequency,
        and weights are normalized to a maximum of 1.
        """

        if weight_sigma is None:
            weight_sigma = 25. * self.del_nu   # Default spread (10 spectral resolution elements)
        weights = np.exp(-0.5 * ((self.x_to_fit - self.nu_rest) / weight_sigma) ** 2)
        # Normalize weights to have a maximum value of 1
        weights /= weights.max()
        self.weights = weights

    def _make_initial_guess(self,
                            input_data,
                            input_x,
                            fit_type='Moffat',
                            save_guess=False,
                            outfile='initial_guess.sav'):
        """
        Make initial guess to fit with LMFIT

        Parameters
        ----------
        fit_type : string
            The type of line shape used to generate the initial guess. A value of 'All' will result in an error.
        save_guess : bool; default = False
            Set to True to save initial guess
        outfile : string; default = initial_guess.sav'
            The name of the output file save file. the file extension '.sav' will be added.
        """
        pars = model[fit_type]['lmfit_func'].guess(input_data, x=input_x)
        self.lmfit_initial_guess_model = model[fit_type]['lmfit_func'].fit(input_data,
                                                                           pars,
                                                                           x=input_x,
                                                                           scale_covar=False)

        if save_guess:
            print('Saving initial model results to ', outfile)
            print(self.lmfit_initial_guess_model.fit_report())
            save_modelresult(self.lmfit_initial_guess_model, outfile)
#######################
# The end of Spectrum()
#######################


#############################
# The start of SpectralCube()
#############################
class SpectralCube:
    """
    Class for handling spectral cube data from ALMA
    """
    def __init__(self, cubepath, nu_rest_str, mask=None):
        """
        Initialize a SpectralCube object with a FITS image path.

        Parameters
        ----------
        cubepath: string
            The absolute path to FITS cube file.
        mask: boolean array; default = None
            Mask array where values set to True will not be analyzed and skipped entirely.
        nu_rest_str: str; default = None
            Rest frequency identifier.
        """

        # Using a context manager to open the FITS file
        with fits.open(cubepath) as fits_file:
            hdr = fits_file[0].header
            self.n_chan = hdr['NAXIS3']  # Number of frequency channels.
            self.i0 = hdr['CRPIX3']      # Reference pixel in the spectral axis.
            self.del_nu = hdr['CDELT3']  # Spectral resolution of FITS cube.
            self.nu_i = hdr['CRVAL3']    # Starting frequency of FITS cube.
            self.ni = hdr['NAXIS1']      # Number of spatial pixels in x direction.
            self.nj = hdr['NAXIS2']      # Number of spatial pixels in y direction.
            self.data = fits_file[0].data[0, :, :, :]  # Loading the data

            self.epoch = Time(hdr['DATE-OBS'], scale='utc')

            # Beam parameters from the image header
            self.bmaj_deg = hdr['BMAJ']  # Semi-major axis of the restoring beam in degrees
            self.bmaj_arcsec = hdr['BMAJ'] * 3600.0 # Semi-major axis of the restoring beam in arcsec
            self.bmin_deg = hdr['BMIN']  # Semi-minor axis of the restoring beam in degrees
            self.bmin_arcsec = hdr['BMIN'] * 3600.0 # Semi-minor axis of the restoring beam in arcsec

            self.bpa_deg = hdr['BPA']    # Position angle of the restoring beam in degrees

            self.pixscale_x_arcsec = abs(hdr["CDELT1"])  # deg / pixel
            self.pixscale_y_arcsec = abs(hdr["CDELT2"])  # deg / pixel
            self.pixscale_x_arcsec = abs(hdr["CDELT1"]) * 3600.0  # arcsec / pixel
            self.pixscale_y_arcsec = abs(hdr["CDELT2"]) * 3600.0  # arcsec / pixel

        # Adjust the reference pixel index.
        self.i0 = self.i0 - 1 if self.i0 < 0 else self.i0 + 1

        # Construct the frequency array in GHz.
        self.nu_arr = ((np.arange(self.n_chan) - self.i0) * self.del_nu + self.nu_i) / 1.e9

        # Set up the mask if provided; default to False mask.
        self.mask = mask if mask is not None else np.zeros((self.ni, self.nj), dtype=bool)

        # Set rest frequency, raising an error if not provided or invalid.
        self.nu_rest = get_rest_frequency(nu_rest_str)

        # Initialize arrays for fit parameters.
        self.fit_peak_ampltitude = np.zeros_like(self.mask, dtype=np.float32)
        self.fit_peak_center_nu = np.zeros_like(self.mask, dtype=np.float32)
        self.fit_peak_sigma = np.zeros_like(self.mask, dtype=np.float32)
        self.fit_beta = np.zeros_like(self.mask, dtype=np.float32)
        self.center_from_data = np.zeros_like(self.mask, dtype=np.float32)
        self.no_fit_pixels = None  # For storing pixels indexes with invalid fits.

        # Initialize arrays Doppler wind velocity.
        self.Doppler_winds = np.zeros_like(self.mask, dtype=np.float32)
        self.Doppler_winds_err = np.zeros_like(self.mask, dtype=np.float32)

        self.winds_from_data = np.zeros_like(self.mask, dtype=np.float32)

    def reshape_data_cube(self, nu_low, nu_high):
        """
        Reshape the data cube by reducing the spectral axis to the range
        specified by nu_low and nu_high.

        Parameters:
        ----------
        nu_low : float
            Lower bound of the frequency range to retain.
        nu_high : float
            Upper bound of the frequency range to retain.
        """
        spectral_fit_mask = (self.nu_arr >= nu_low) & (self.nu_arr <= nu_high)
        self.nu_arr = self.nu_arr[spectral_fit_mask]
        self.data = self.data[spectral_fit_mask, :, :]

        if self.data.size == 0:
            raise ValueError("The specified frequency range results in an empty data cube.")

    def fit_cube_spectra(self,
                         fit_type='Moffat',
                         snr_threshold=5.0,
                         guess_smooth_frac=0.05,
                         bound_fit=False,
                         bound_velocity_maximum=None,
                         use_weights=False,
                         weight_sigma=None,
                         plot_pixel_fit=False,
                         make_pixel_plot_gif=False,
                         gif_outfile='pixel_fits.gif',
                         indices_list=None):
        """
        Calculate the best fit for the spectra at each specified pixel.

        Parameters
        ----------
        fit_type: str
            The type of line shape used to fit the spectra. Options include 'Moffat', 'Gaussian', etc.
        snr_threshold: float; default=3.0
            The estimated signal to noise ratio for a spectrum to be fitted.
        guess_smooth_frac: float; default=0.01
            The window length of the fractional percentage of data to fit to smooth for the inital guess.
        bound_fit: bool; default=True
            If True, apply bounds to the fit parameters.
        bound_velocity_maximum: folat; default=1e5
            The maximum velocity to extend the fitting to away from the rest frequency in m/s.
        use_weights: bool,
            If True, apply weights to the fitting process.
        weight_sigma: float; default=None,
            The FWHM of the Guassian weights applied to the fitted data.
        plot_pixel_fit: bool, optional
            If True, plot individual pixel fits.
        make_pixel_plot_gif: bool, optional
            If True, generate a GIF of the pixel fits.
        gif_outfile: str, optional
            The output filename for the GIF.
        indices_list: list of tuples, optional
            List of (i, j) pixel indices to limit the fitting process.
        moreno:; bool
            using this data set
        """

        error_pixels = []

        # Default to unmasked indices if no limit is provided
        if indices_list is None:
            unmasked_indices = np.argwhere(~self.mask)
            print(f"Fitting all unmasked pixels in spectral cube... 0%", end="", flush=True)
        else:
            unmasked_indices = np.array(indices_list)
            print(f"Fitting specified pixels in indices list.", end="", flush=True)

        total_unmasked = len(unmasked_indices)

        if total_unmasked == 0:
            print("No pixels to fit.")
            return

        # Process specified indices
        for count, (i, j) in enumerate(unmasked_indices):
            # Skip if the pixel has intensities that are all NaN
            if np.all(np.isnan(self.data[:, i, j])):
                continue
            else:
                # Instantiate Spectrum
                spec_data = SpectrumData(
                    intensity=self.data[:, i, j],
                    frequency=self.nu_arr,
                    nu_rest=self.nu_rest,
                    i=i,
                    j=j
                )

                spectrum = Spectrum(spec_data, snr_threshold=snr_threshold)

                try:
                    # Fit the profile using the specified fit type
                    results = spectrum.fit_profile(fit_type=fit_type,
                                                   guess_smooth_frac=guess_smooth_frac,
                                                   bound_fit=bound_fit,
                                                   bound_velocity_maximum=bound_velocity_maximum,
                                                   use_weights=use_weights,
                                                   weight_sigma=weight_sigma,
                                                   plot_pixel_fit=plot_pixel_fit
                                                   )

                    # Extract the center frequency from the fit result
                    if results is not None:
                        self.fit_peak_ampltitude[i, j] = results.amplitude
                        self.fit_peak_center_nu[i, j] = results.center
                        self.fit_peak_sigma[i, j] = results.sigma
                        self.fit_beta[i, j] = results.beta

                        self.center_from_data[i, j] = (
                            spectrum.nu_at_Imax if spectrum.nu_at_Imax is not None else 0
                        )

                        #print(results[1], self.fit_center_diff_from_data[i, j], self.center_from_data[i, j])

                except ValueError as e:
                    # Append pixel location to the error_pixels list
                    error_pixels.append((i, j))

                    # Set results to zero for that pixel
                    self.fit_peak_ampltitude[i, j] = 0
                    self.fit_peak_center_nu[i, j] = 0
                    self.fit_peak_sigma[i, j] = 0
                    self.fit_beta[i, j] = 0

                    self.center_from_data[i, j] = 0

            if indices_list is None:
                # Update the progress every 10% completion
                progress = (count + 1) / total_unmasked
                if progress >= 0.1 and (count + 1) % (total_unmasked // 10) == 0:
                    print(f" {int(progress * 100)}%", end="", flush=True)

        print("\nCompleted fitting.")

        # Create a GIF from diagnostic plots
        if make_pixel_plot_gif:
            if gif_outfile:
                create_fit_gif(gif_outfile=gif_outfile)
            else:
                create_fit_gif(gif_outfile='pixel_fits.gif')

        self.no_fit_pixels = error_pixels

    def make_Doppler_velocity_arrays(self):
        """
        Generate Doppler velocity arrays for each pixel in the spectral cube.

        This method computes the Doppler velocity for each pixel based on the peak center frequency
        (`fit_peak_center_nu`) and the frequency standard deviation (`fit_peak_sigma`). The results
        are stored in the `Doppler_winds` array. Pixels that are masked or have a peak center frequency
        equal to zero are skipped.

        Notes:
        - A `mask` value of `True` indicates that the pixel is masked and should be skipped.
        - If the `fit_peak_center_nu` value is zero for a pixel, Doppler velocity is not calculated for it.
        - The `mask` is updated to `NaN` for skipped pixels.

        Returns
        -------
        None
            Results are stored in the `Doppler_winds` array attribute.

        Updates Attributes
        ------------------
        Doppler_winds : ndarray
            The calculated Doppler velocities for valid pixels.
        mask : ndarray
            Updated mask where skipped pixels are set to NaN.

        """
        # Iterate over each pixel and compute Doppler velocity if not masked
        for i in range(self.ni):
            for j in range(self.nj):
                # Check if the pixel is masked
                if self.mask[i, j]:
                    self.mask[i, j] = np.nan
                    continue  # Skip if pixel is masked or set to True

                # Skip calculation if nu_cen is zero
                if self.fit_peak_center_nu[i, j] != 0:
                    self.Doppler_winds[i, j], _ = calc_Doppler_velocity(
                        nu_cen=self.fit_peak_center_nu[i, j],
                        nu_sig=self.fit_peak_sigma[i, j],
                        nu_rest=self.nu_rest
                    )

                    self.winds_from_data[i, j], _ = calc_Doppler_velocity(
                        nu_cen=self.center_from_data[i, j],
                        nu_sig=self.fit_peak_sigma[i, j],
                        nu_rest=self.nu_rest
                    )

                    #print(i, j, self.Doppler_winds[i, j], self.winds_from_data[i, j], self.winds_diff_data[i, j])

    def make_pickle_results(self, pickle_filename="doppler_winds.pkl"):
        # Validate that pickle_filename is a string
        if not isinstance(pickle_filename, str):
            raise TypeError("The pickle file name must be a string.")

        # Prepare the data dictionary
        data = {
            "Doppler_winds": self.Doppler_winds,
            "bmaj": self.bmaj,
            "bmin": self.bmin,
            "bpa": self.bpa,
        }

        # Write data to the pickle file
        try:
            with open(pickle_filename, "wb") as file:
                pickle.dump(data, file)

            # Verify the pickle file exists
            if not os.path.exists(pickle_filename):
                raise FileNotFoundError(f"Failed to create the pickle file: {pickle_filename}")

            print(f"Pickle file '{pickle_filename}' successfully created.")

        except Exception as e:
            raise IOError(f"An error occurred while creating the pickle file: {e}")

#############################
# The end of SpectralCube()
#############################


def get_rest_frequency(nu_rest_name):
    """
    Retrieve the rest frequency value by name from the REST_FREQUENCIES dictionary.
    """
    try:
        return REST_FREQUENCIES[nu_rest_name]
    except KeyError:
        raise ValueError(f"Unknown rest frequency name: {nu_rest_name}")


def smooth_data(data_to_smooth, fraction=0.01, polyorder=3):
    """
    Apply Savitzky-Golay filtering to check the signal's RMS and flag significant signals.

    Parameters
    ----------
    data_to_smooth: ndarray
        The data array to apply the filter to.
    fraction: float, default=0.01 or 1%
        Fraction of the total number of points in `self.I_nu_arr` to use as the smoothing filter length.
        Default is 0.01 (1% of the total points).
    polyorder: int, default=3
        The polynomial order for the Savitzky-Golay filter.
    """
    total_points = len(data_to_smooth)
    filter_length = max(3, int(fraction * total_points) | 1)  # Ensure odd integer >= 3
    smoothed_data = savgol_filter(data_to_smooth, filter_length, polyorder)
    return smoothed_data, filter_length


def calc_Doppler_velocity(nu_cen, nu_sig, nu_rest):
    """
    Function to calculate the measured Doppler wind velocity from the spectral cube
    and the selected molecular transition. Doppler velocity is defined as the velocity
    (positive or negative) from the shifted line transition from the rest frequency. This
    velocity does not include viewing geometry, line of sight corrections, or planetary
    body solid rotation.

    Parameters
    ----------
    nu_cen: float;
        The calculated frequency in GHz center of the spectral fitted profile.
    nu_sig: float
        The standard deviation of the calculated spectral fitted profile.
    nu_rest: float
        The rest frequency in GHz to compare nu_cen to calculate Doppler velocity.

    Returns
    -------
    Doppler_velocity : float
          The calculated Doppler velocity in m/s.
    Doppler_velocity_error : float
          The Doppler velocity error, calculated through error propagation, in m/s.
    """
    a = c.value / nu_rest
    Doppler_velocity = c.value * (1. - (nu_cen / nu_rest))
    Doppler_velocity_error = np.abs(a * nu_sig)

    return Doppler_velocity, Doppler_velocity_error

def generate_diagnostic_plot(
    nu_arr,
    I_nu_arr,
    filtered_snr_data,
    filtered_snr_data_max_value,
    x_to_fit,
    y_to_fit,
    y_for_initial_guess,
    lmfit_initial_guess_model,
    lmfit_fit_result_model,
    nu_low_bound,
    nu_high_bound,
    weights,
    nu_at_Imax,
    Imax_value,
    snr,
    rms,
    i,
    j,
    plot_dir='./plots',
    outfile=None
):
    """
    Generate a diagnostic plot for the fitted spectrum.
    """

    if outfile is None:
        outfile = os.path.join(plot_dir, f'diagnostic_fitting_plot_i{i:04d}_j{j:04d}.png')

    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=True)

    # -------------------------
    # Upper left
    # -------------------------
    ax1 = axes[0, 0]
    ax1.plot(nu_arr, I_nu_arr, 'k.', label="Original Data")
    ax1.plot(nu_arr, filtered_snr_data, 'g.', label="Filtered Data")

    if nu_low_bound and nu_high_bound:
        ax1.axvline(nu_low_bound, color='grey', linestyle='-.')
        ax1.axvline(nu_high_bound, color='grey', linestyle='--')

    ax1.set_title(f"Pixel ({i}, {j})")
    ax1.text(0.02, 0.95, f"SNR: {snr:.2f}", transform=ax1.transAxes)
    ax1.text(0.02, 0.90, f"RMS: {rms:.2f}", transform=ax1.transAxes)
    ax1.text(0.02, 0.85, f"Threshold: {rms * snr:.2f}", transform=ax1.transAxes)
    ax1.text(0.02, 0.80, f"Max: {filtered_snr_data_max_value:.2f}", transform=ax1.transAxes)

    ax1.legend()

    # -------------------------
    # Upper right (initial guess)
    # -------------------------
    ax2 = axes[0, 1]
    ax2.plot(x_to_fit, y_for_initial_guess, '.', color='grey')

    if lmfit_initial_guess_model is not None:
        ax2.plot(x_to_fit,
                 lmfit_initial_guess_model.eval(x=x_to_fit),
                 '--', color='gold')

    ax2.set_title("Initial Guess")

    # -------------------------
    # Lower left (weights)
    # -------------------------
    ax3 = axes[1, 0]
    if weights is not None:
        ax3.plot(x_to_fit, weights)
        ax3.set_title("Weights")
    else:
        ax3.set_title("No Weights")
        ax3.axis('off')

    # -------------------------
    # Lower right (fit)
    # -------------------------
    ax4 = axes[1, 1]
    ax4.plot(x_to_fit, y_to_fit, '.', color='grey')

    if lmfit_fit_result_model is not None:
        ax4.plot(x_to_fit,
                 lmfit_fit_result_model.best_fit,
                 '--', color='purple')

    ax4.set_title("Fit")

    if nu_at_Imax is not None:
        ax4.text(0.02, 0.30, f"nu_at_Imax: {nu_at_Imax:.2f}", transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)


def create_fit_gif(plot_dir='./plots',
                   gif_outfile='fit_diagnostic_plots.gif',
                   duration=0.5,
                   delete_pngs=True):
    """
    Create a GIF from diagnostic plot images located in the specified plot directory.

    Parameters
    ----------
    plot_dir : str, optional
        The directory where diagnostic plots are saved. Default is './plots'.
    gif_outfile : str, optional
        The output file name for the generated GIF. Default is 'fit_diagnostic_plots.gif'.
    duration : float, optional
        Duration for each frame in the GIF, in seconds. Default is 0.5.
    delete_pngs : bool, optional
        If True, delete the individual PNG files after creating the GIF. Default is True.
    """
    try:
        # Ensure the plot directory exists
        if not os.path.exists(plot_dir):
            print(f"The specified plot directory does not exist: {plot_dir}")
            return

        # Gather diagnostic fitting plot files
        diagnostic_files = [file for file in os.listdir(plot_dir)
                            if file.startswith('diagnostic_fitting_plot_') and file.endswith('.png')]

        if not diagnostic_files:
            print(f"No diagnostic plot PNG files found in the directory: {plot_dir}")
            return

        diagnostic_files.sort()
        png_files = [os.path.join(plot_dir, file) for file in diagnostic_files]
        images = [imageio.imread(file) for file in png_files]
        imageio.mimsave(gif_outfile, images, duration=duration)
        print(f"GIF created: {gif_outfile}")

        # Optionally delete the PNG files
        if delete_pngs:
            for png_file in png_files:
                os.remove(png_file)
            print("Individual PNG files deleted.")

    except Exception as e:
        print(f"An error occurred while creating the GIF: {e}")

