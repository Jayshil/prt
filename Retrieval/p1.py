import numpy as np
import matplotlib.pyplot as plt
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.spectral_model import SpectralModel
from petitRADTRANS.planet import Planet
from scipy.stats import binned_statistic
import dynesty
from dynesty.utils import resample_equal
from juliet.utils import get_quantiles
from scipy import stats
import astropy.units as u
from krithika import plotstyles
import pickle
import seaborn as sns
import os
from pathlib import Path
from multiprocessing import Pool
import contextlib

import multiprocessing
multiprocessing.set_start_method('fork')

from matplotlib import rcParams
rcParams['figure.dpi'] = 300

p1 = os.getcwd()
pout = p1 + '/Retrieval/Analysis/Isothermal/'
if not Path(pout).exists():
    os.mkdir(pout)

# Loading the planet
toi396b = Planet.get(name='HR 858 b')

# Loading the data
wav_f322, dep_f322, dep_err_f322 = np.loadtxt(p1 + '/GenTSO/Data/toi396b_transit_spectrum_isothermal_f322w2_R250.txt', usecols=(0,1,2), unpack=True)
wav_f444, dep_f444, dep_err_f444 = np.loadtxt(p1 + '/GenTSO/Data/toi396b_transit_spectrum_isothermal_f444w_R250.txt', usecols=(0,1,2), unpack=True)

# Loading the transmission function
## For the F322W2 filter
lam_f322, res_f322 = np.loadtxt(p1 + '/Retrieval/Data/JWST_NIRCam.F322W2.dat', usecols=(0,1), unpack=True)
lam_f322 = ( lam_f322 * u.AA ).to(u.micron).value
## For the F444W filter
lam_f444, res_f444 = np.loadtxt(p1 + '/Retrieval/Data/JWST_NIRCam.F444W.dat', usecols=(0,1), unpack=True)
lam_f444 = ( lam_f444 * u.AA ).to(u.micron).value

specmodel = SpectralModel(
    # Radtrans parameters
    pressures=np.logspace(-6, 2, 100),
    line_species=[
        'CO2',
        'CH4'
    ],
    rayleigh_species=['H2', 'He'],
    gas_continuum_contributors=['H2--H2', 'H2--He'],
    wavelength_boundaries=[1, 6],
    
    # Model parameters
    ## Planet parameters
    planet_radius=toi396b.radius,
    reference_gravity=toi396b.reference_gravity,
    reference_pressure=toi396b.reference_pressure,
    
    ## Temperature profile parameters
    temperature=1000,  # isothermal temperature profile
    
    # Mass fractions
    imposed_mass_fractions={  # these can also be arrays of the same size as pressures
        'CO2': 10e-6,
        'CH4': 10e-6
    },
    filling_species={  # automatically fill the atmosphere with H2 and He, such that the sum of MMRs is equal to 1 and H2/He = 37/12
        'H2': 3.096,
        'He': 1.
    },

    # Opacity mode (line-by-line in this case)
    line_opacity_mode='c-k'

)

def gaussian_log_likelihood(residuals, variances):
    taus = 1. / variances
    return -0.5 * (len(residuals) * np.log(2*np.pi) + np.sum(-np.log(taus.astype(float)) + taus * (residuals**2)))

def evaluate_model(wav, dep, dep_err, spec, log_co2, log_ch4, temp, offset, wav_res=None, res_func=None):
    # Let's first modify the mass fractions
    spec.model_parameters['imposed_mass_fractions']['CO2'] = 10**log_co2
    spec.model_parameters['imposed_mass_fractions']['CH4'] = 10**log_ch4
    
    # And updating the temperature for an isothermal profile
    spec.model_parameters['temperature'] = temp

    # Calculating the model spectrum
    wave_model, transit_radii_model = spec.calculate_spectrum(
        mode='transmission'
    )
    
    ## Converting units
    wave_model = ( wave_model[0,:] * u.cm ).to(u.micron).value
    dep_model = ( ( transit_radii_model[0,:] / toi396b.star_radius )**2 ) * 1e6  # converting to ppm

    # --- This model would be at the native resolution of petitRADTRANS, so we need to bin it to the resolution of the data ---
    ## Wavelength boundaries
    wav_diff = np.hstack([ np.diff(wav)[0], np.diff(wav) ])  # calculating the wavelength differences, and adding the last one to keep the same size
    wav_st, wav_end = wav - wav_diff/2, wav + wav_diff/2  # calculating the wavelength boundaries for each data point

    ## Binning the model spectrum to the resolution of the data (vectorized)
    bin_edges = np.append(wav_st, wav_end[-1])
    dep_model_binned, _, _ = binned_statistic(wave_model, dep_model, statistic='mean', bins=bin_edges)

    if res_func is not None and wav_res is not None:
        ## Binning the resolution function to the same wavelength grid as the data (vectorized)
        res_func_binned, _, _ = binned_statistic(wav_res, res_func, statistic='mean', bins=bin_edges)
        ## Multiplying the response function with the binned model spectrum
        dep_model_binned *= res_func_binned

    ## Apply offset
    transit_depth_model = dep_model_binned + offset

    # Calculating the log-likelihood
    residuals = dep - transit_depth_model
    log_likelihood = gaussian_log_likelihood(residuals, dep_err**2)
    
    return dep_model_binned, log_likelihood

# ------------------------------------------------------------------
#
# Defining the data in the format that we will use for the retrieval
#
# ------------------------------------------------------------------
wav, dep, dep_err = {}, {}, {}
wav['F322W2'], dep['F322W2'], dep_err['F322W2'] = wav_f322, dep_f322, dep_err_f322
wav['F444W'], dep['F444W'], dep_err['F444W'] = wav_f444, dep_f444, dep_err_f444


# ------------------------------------------------------------------
#
#                        Defining the priors
#
# ------------------------------------------------------------------
## We will basically fit for four parameters: the log of the mass fractions of CO2 and CH4, and the temperature of the isothermal profile.
#  We will also add an offset parameter to account for any potential systematic offsets in the data.

par = ['logCO2', 'logCH4', 'temp', 'offset_F322W2', 'offset_F444W']
dist = ['uniform', 'uniform', 'uniform', 'fixed', 'uniform']
hypers = [[-10, -1], [-10, -1], [500, 2000], 0., [-100, 100]]

## Prior transform
def uniform(t, a, b):
    return (b-a)*t + a
def stand(a, loc, scale):
    return (a-loc)/scale

# Saving prior file
f11 = open(pout + '/priors.dat', 'w')
for i in range(len(par)):
    f11.write(par[i] + '\t' + dist[i] + '\t' + str(hypers[i]) + '\n')
f11.close()

# First leave the fixed parameters
free_params, free_dists, free_hypers = [], [], []
fixed_params, fixed_vals = [], []
for i in range(len(par)):
    if dist[i] != 'fixed':
        free_params.append(par[i])
        free_dists.append(dist[i])
        free_hypers.append(hypers[i])
    else:
        fixed_params.append(par[i])
        fixed_vals.append(hypers[i])

# Prior cube for only free parameters
def prior_transform(ux):
    x = np.array(ux)
    for i in range(len(free_params)):
        if free_dists[i] == 'uniform':
            x[i] = uniform(ux[i], free_hypers[i][0], free_hypers[i][1])
        elif free_dists[i] == 'normal':
            x[i] = stats.norm.ppf(ux[i], loc=free_hypers[i][0], scale=free_hypers[i][1])
        elif free_dists[i] == 'truncatednormal':
            x[i] = stats.truncnorm.ppf(ux[i], a=stand(free_hypers[i][2], free_hypers[i][0], free_hypers[i][1]), b=stand(free_hypers[i][3], free_hypers[i][0], free_hypers[i][1]), loc=free_hypers[i][0], scale=free_hypers[i][1])
        elif free_dists[i] == 'loguniform':
            x[i] = stats.loguniform.ppf(ux[i], a=free_hypers[i][0], b=free_hypers[i][1])
        else:
            raise Exception('Please use proper distribution!')
    return x

# ------------------------------------------------------------------
#
#            Defining the log-likelihood function
#
# ------------------------------------------------------------------
nthreads = multiprocessing.cpu_count()

def loglike(x):
    global wav, dep, dep_err, specmodel, par, free_params, fixed_params, fixed_vals

    # Saving values of parameters in a dictionary

    parameters = {}
    for p in range(len(free_params)):
        parameters[free_params[p]] = x[p]
    for p in range(len(fixed_params)):
        parameters[fixed_params[p]] = fixed_vals[p]

    # And computing log-likelihood
    log_like = 0
    # First estimating the model
    for ins in ['F322W2', 'F444W']:
        _, loglikelihood = evaluate_model(wav=wav[ins], dep=dep[ins], dep_err=dep_err[ins], spec=specmodel,\
                                          log_co2=parameters['logCO2'], log_ch4=parameters['logCH4'], temp=parameters['temp'],\
                                          offset=parameters['offset_' + ins], wav_res=None, res_func=None)
        
        log_like = log_like + loglikelihood
    
    return log_like

# ------------------------------------------------------------------
#
#                      Sampling with dynesty
#
# ------------------------------------------------------------------

out_files = Path(pout + '/_dynesty_DNS_posteriors.pkl')
## Only start sampler if dynesty output files are not detected
## Otherwise, just load them
if not out_files.exists():
    with contextlib.closing(Pool(processes=nthreads-1)) as executor:
        dsampler = dynesty.DynamicNestedSampler(loglikelihood=loglike, prior_transform=prior_transform,\
            ndim=len(free_params), nlive=500, bound='single', sample='rwalk', pool=executor, queue_size=nthreads)
        dsampler.run_nested()
    dres = dsampler.results

    weights = np.exp(dres['logwt'] - dres['logz'][-1])
    posterior_samples = resample_equal(dres.samples, weights)

    f22 = open(pout + '/posteriors.dat', 'w')
    post_samps = {}
    post_samps['posterior_samples'] = {}
    for i in range(len(free_params)):
        post_samps['posterior_samples'][free_params[i]] = posterior_samples[:, i]
        qua = get_quantiles(posterior_samples[:, i])
        f22.write(free_params[i] + '\t' + str(qua[0]) + '\t' + str(qua[1]-qua[0]) + '\t' + str(qua[0]-qua[2]) + '\n')
    f22.close()

    # logZ
    post_samps['lnZ'] = dres.logz
    post_samps['lnZ_err'] = dres.logzerr

    # Dumping a pickle
    pickle.dump(post_samps,open(pout + '/_dynesty_DNS_posteriors.pkl','wb'))
else:
    print('>>>> --- Dynesty sampler files are detected!!!')
    print('>>>> --- Loading them...')
    post_samps = pickle.load(open(pout + '/_dynesty_DNS_posteriors.pkl', 'rb'))
    print('>>>> --- Done!')