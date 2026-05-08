import numpy as np
import matplotlib.pyplot as plt
import gen_tso.pandeia_io as jwst
import pickle
import os
#plt.ion()

instrument = 'f322w2'

with open(os.getcwd() + f'/GenTSO/Data/tso_transit_iso_HR858b_nircam_lw_grism_{instrument}.pickle', 'rb') as f:
    tso = pickle.load(f)

# Draw a simulated transit spectrum at selected resolution
# - Set n_obs to simulate repeated observations to improve the S/N
# - Set noiseless=True to simulate spectra with no scatter noise
obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
    tso, resolution=250.0, n_obs=1,
)


fig, axs = plt.subplots(figsize=(16/1.5, 9/1.5))
axs.plot(tso['wl'], tso['depth_spectrum']*1e6, c='salmon', label='depth at instrumental resolution')
axs.errorbar(obs_wl, obs_depth*1e6, xerr=band_widths, yerr=obs_error*1e6, fmt='o', ms=5, color='xkcd:blue', mfc=(1,1,1,0.85), label='simulated (noised up) transit spectrum')

axs.set_ylim([100, 650])

axs.set_xlabel('Wavelength [micron]')
axs.set_ylabel('Transit depth [ppm]')

plt.legend(loc='best')
plt.show()

# -------- Saving the data --------
np.savetxt(os.getcwd() + f'/GenTSO/Data/toi396b_transit_spectrum_isothermal_{instrument}_R250.txt', np.array([obs_wl, band_widths, obs_depth*1e6, obs_error*1e6]).T, header='Wavelength [micron], Band width [micron], Transit depth [ppm], Transit depth error [ppm]')