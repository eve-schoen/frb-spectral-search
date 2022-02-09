import numpy as np
import matplotlib.pyplot as plt
from baseband_analysis.core.sampling import _upchannel as upchannel 
#this version allows me to upchannel a piece rather than everything, saving lots of time

import scipy.stats

from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#imports just for get_smooth_matched_filter_new
from matplotlib.pyplot import *
from scipy.stats import median_absolute_deviation
from baseband_analysis.analysis.snr import get_snr
from baseband_analysis.core.sampling import fill_waterfall, _scrunch, downsample_power_gaussian, clip
from baseband_analysis.core.signal import get_main_peak_lim, tiedbeam_baseband_to_power, get_weights
from common_utils import delay_across_the_band 
def calculate_noise_simple(ww, time_range, on_range, valid_channels,scale =False): 
    #Purpose: calculates noise average and standard deviation based off of off-pulse data
    #looks at chunks of time equal to the on-burst and processes them identically, 
    #ww has waterfall and rfi but this upchannelizes and deripples
    
    noise_ranges =[] 
    l_on = on_range[1]-on_range[0]
    bottom = on_range[0]-2*l_on

    while bottom > time_range[0]:
        noise_ranges.append([bottom, bottom+l_on])
        bottom -= l_on
    top = on_range[1]+2*l_on+1
    while top < time_range[1]-1:
        noise_ranges.append([top-l_on, top])
        top += l_on

    my_freq_id = np.linspace(0,1023, 1024) 

    #initialize list of spectrums
    x= noise_ranges[0]
    spec, frbindexes, frbchan_id_upchan = upchannel(ww[:,:,x[0]:x[1]], my_freq_id)
    spec1d = np.sum(np.sum(np.abs(spec)**2,axis = 0),axis = 0)
    fix = np.array([0.67461854, 0.72345924, 0.8339084 , 0.96487784, 1.0780604 ,
       1.1574216 , 1.2020986 , 1.2204636 , 1.2236487 , 1.2168874 ,
       1.1857561 , 1.1230892 , 1.0274547 , 0.9002419 , 0.7799086 ,
       0.68808776])
    big_fix = np.tile(fix, len(spec1d)//16)
    spec1d = np.multiply(spec1d, 1/big_fix)
    n_range_specs = spec1d
    count = 1

    for x in noise_ranges[1:50]:   
        if count%5 ==0:
            print("spec" + str(count) +" out of " + str(len(noise_ranges)-1))
        count +=1
        spec, frbindexes, frbchan_id_upchan = upchannel(ww[:,:,x[0]:x[1]], my_freq_id)
        spec1d = np.sum(np.sum(np.abs(spec)**2,axis = 0),axis = 0)
        spec1d = np.multiply(spec1d, 1/big_fix)
        added = np.append(n_range_specs, spec1d)
        n_range_specs = np.reshape(added, [-1, 1024*16]) #use -1 so it fill in that dimension
    t_sys = np.nanmean(n_range_specs, 0)
    scale_fact = 1
    my_freq_id = np.arange(1024)   
    frbspec, frbindexes, frbchan_id_upchan = upchannel(ww[:,:,on_range[0]:on_range[1]], my_freq_id)
    ospec = np.sum(np.sum(np.abs(frbspec)**2,axis = 0),axis = 0)
    ospec = np.multiply(ospec, 1/big_fix)
    scale_fact_calc = ospec/t_sys
    scale2  = np.ndarray([])
    for x in scale_fact_calc:
        if x < 1:
            scale2 = np.append(scale2, 1)
        else:
            scale2 = np.append(scale2, x)          
    if scale:
        scale_fact = np.nanmean(scale2)
    print('Error bar scaling factor: ' + str(scale_fact))
    spec_ebar = scipy.stats.median_abs_deviation(n_range_specs, axis=0, nan_policy = 'omit')*scale_fact
    return n_range_specs, t_sys, spec_ebar, scale_fact_calc

def plot_spectra_seperate_polarizations(ww, start_bin, end_bin, valid_channels, time_range, DM, noise_spec, noise_std):
    # Purpose: plotting the onburst spectrum with the Macquart region for expected 21 cm line off of DM
    #shows spectrum after processing (upchannelization and noise subtracting and derippling compared to original spectrum)
    #ww is the data rid of rfi and filled already
    my_freq_id = np.arange(1024)   
    frbspec, frbindexes, frbchan_id_upchan = upchannel(ww[:,:,start_bin:end_bin], my_freq_id)
    spectras = np.empty(shape=(2, 1024*16))
    spectras[1, :] = fix_rippling(np.sum(np.abs(frbspec)**2,axis = 1))
    spectras[0,:] = fix_rippling(np.sum(np.abs(frbspec)**2,axis = 0))
    print(len(spectra))
    print(len(spectra[0]))
    spectra= np.array([fix_rippling_and_subtract_nosie(spectra[0], noise_spec), fix_rippling_and_subtract_nosie(spectra[1], noise_spec)])
    print(len(spectra))
    print(len(spectra[0]))
    return frbspec, frbindexes, spectra, noise_std

def fix_rippling(spectra):
    fix = np.array([0.67461854, 0.72345924, 0.8339084 , 0.96487784, 1.0780604 ,
       1.1574216 , 1.2020986 , 1.2204636 , 1.2236487 , 1.2168874 ,
       1.1857561 , 1.1230892 , 1.0274547 , 0.9002419 , 0.7799086 ,
       0.68808776])  
    big_fix = np.tile(fix, len(spectra)//16)
    spectra = np.multiply(spectra, 1/big_fix)
    return spectra
    
def get_smooth_matched_filter_new(
        data,
        DM = None,
        DM_range = None,
        downsampling_factor = None,
        lim = None,
        diagnostic_plots = False,
        full_output = True,
        floor_level=0.1,
        subtract_dc = True,
        valid_time_range=None,
        flag_bad_rfi_chans=True,
        return_dedisp_array=True):
    # 1) run get_snr to find what valid_span_power_bins and downsampling_factor and DM_clip is.
    _, _, power_temp, _, _, _, valid_span_power_bins, DM_clip, downsampling_factor = get_snr(
            data,
            diagnostic_plots=False,
            downsample = downsampling_factor,
            DM = DM,
            DM_range = DM_range,
            spectrum_lim=False,
            return_full = True,
            fill_missing_time = False)
    start_power_bins_temp, end_power_bins_temp = get_main_peak_lim(power_temp,
            floor_level=floor_level, 
            diagnostic_plots=False,
            normalize_profile=True) # calculate start and end here just to make sure the clipper does not lose the pulse.
    pulse_span_power_bins = np.array([start_power_bins_temp,end_power_bins_temp])
    pulse_span_bottom = data['time0']['ctime'][-1] + 2.56e-6 * pulse_span_power_bins  
    print('The arguments were:')
    print(f'get_smooth_matched_filter(data,DM = {DM_clip},downsampling_factor = {downsampling_factor})')
    figure()
    imshow(power_temp,aspect = 'auto')
    # 2) find valid time range and clip the BBData object, by looking at the corners of the "trapezoid" and cutting the biggest rectangle out of it possible.
    # bottom corners taken from get_snr
    valid_span_unix_bottom = data['time0']['ctime'][-1] + 2.56e-6 * valid_span_power_bins * downsampling_factor
    # top corners taken from data shape
    dm_delay_sec = delay_across_the_band(DM = data['tiedbeam_baseband'].attrs['DM'],
        freq_high = data.index_map['freq']['centre'][0], 
        freq_low = data.index_map['freq']['centre'][-1])
    valid_span_unix_top = data['time0']['ctime'][0] + dm_delay_sec + np.array([0,data.ntime]) * 2.56e-6
    valid_times = [max(valid_span_unix_bottom[0],valid_span_unix_top[0]),
                  min(valid_span_unix_bottom[1],valid_span_unix_top[1])]
    # translate times to TOA and duration; run the clipper
    toa_400 = np.mean(valid_times) # calculate the center of the data within the valid region
    duration = valid_times[1] - valid_times[0] # calculate duration in seconds
    print(f"sampling.clip(data, dm = {DM_clip:0.3f}, toa_400 = {toa_400}, ref_freq = {data.index_map['freq']['centre'][-1]},duration = {duration}, pad = True")
    data_clipped = clip(data,toa_400 = toa_400, 
                                 duration = duration, 
                                 ref_freq = data.index_map['freq']['centre'][-1],
                                 dm = DM_clip,inplace = False,
                                 pad = True)
    tiedbeam_baseband_to_power(data_clipped,time_downsample_factor = 1,
                                      dedisperse = False,
                                      dm = DM_clip) # post process the data after clipping
    # 3) run get_SNR one more time. This time, we just need time_range_power_bins. 
    _, _, power, _, _, valid_channels, time_range_power_bins, _, downsampling_factor = get_snr(
        data_clipped,
        diagnostic_plots=False,
        downsample = downsampling_factor,
        DM = DM_clip,
        spectrum_lim=False,
        return_full = True,
        fill_missing_time = True)
    figure(3)
    title('Post clip')
    imshow(power,aspect = 'auto')
    if lim is None:
        # Calculate limits and convert from power bins back into frame bins.
        start_power_bins, end_power_bins = get_main_peak_lim(power, 
                floor_level=floor_level, 
                diagnostic_plots=False,
                normalize_profile=True)
        end_power_bins += 1
        start_bin_frames = (time_range_power_bins[0] + start_power_bins) * downsampling_factor # valid region start + start index
        end_bin_frames = (time_range_power_bins[0] + end_power_bins) * downsampling_factor # valid region start + end index
        lim = np.array([start_bin_frames,end_bin_frames])
    lim = np.array(lim)
    # 5) Get matched filter from band and polarization-summed flux, after median subtraction and zeroing outside the window
    w = data_clipped['tiedbeam_baseband'][:] # no need for further dedispersion, because we have used clip()
    w = np.where(np.isnan(w),0,w)
    weights0,offsets0 = get_weights(np.abs(w[:,0])**2,f_id = data_clipped.index_map['freq']['id'],spectrum_lim = False)
    weights1,offsets1 = get_weights(np.abs(w[:,1])**2,f_id = data_clipped.index_map['freq']['id'],spectrum_lim = False)
    w[:,0] /= np.sqrt(weights0)[:,None]
    w[:,1] /= np.sqrt(weights1)[:,None]
    dc_offset = np.mean(w,axis = -1)
    w -= dc_offset[...,None]
    w[~valid_channels] = 0
    _,ww = fill_waterfall(data_clipped,matrix_in = w, write = False)
    # calculate noise and signal temperatures for sensitivity
    flux = np.sum(np.abs(w[valid_channels,...])**2,axis = 0)
    noise_temp = np.nanmedian(flux,axis=  -1)
    signal_temp = np.nanmean(flux[:,lim[0]:lim[1]],axis = -1)
    flux_c = flux.copy() - noise_temp[...,None]
    flux_c[flux_c < 0 ] = 0
    flux_c[:,0:lim[0]] = 0
    flux_c[:,lim[1]:] = 0
    h,_ = downsample_power_gaussian(flux_c,data_clipped,factor = downsampling_factor,upsample = 1)
    h_offset = h.copy() # finite support over the time range
    h[...,0:lim[0]] = 0
    h[...,lim[1]:] = 0
    if valid_time_range is None:
        valid_time_range = [0,ww.shape[-1]] # all frames are valid! yay!
    # 6) Manual clipping to some valid time range
    h = h[...,valid_time_range[0]:valid_time_range[1]]
    ww =ww[:,:,valid_time_range[0]:valid_time_range[1]]
    flux_c = flux_c[:,valid_time_range[0]:valid_time_range[1]]
    flux = flux[:,valid_time_range[0]:valid_time_range[1]]
    lim -= valid_time_range[0] # adjust pulse limits
    # 7) Flag bad channels, RFI channels that seem to not be masked (experimental)
    if flag_bad_rfi_chans:
        chan_spectrum = np.nansum( np.nansum(np.abs(ww)**2,axis=1),axis=-1)
        chan_spectrum_snr = (chan_spectrum - np.nanmedian(chan_spectrum))
        chan_spectrum_snr /= (1.4826*median_absolute_deviation(chan_spectrum,nan_policy='omit'))
        miss_chan_mask = np.where( (chan_spectrum_snr < -1) *(chan_spectrum > 0) )
        ww[miss_chan_mask,:,:] = 0
    if diagnostic_plots is not False:
        def plot_matched_filter_two_pols(ax1,flux,h,pulse_lim,downsampling_factor):
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            rms = median_absolute_deviation(flux, axis = -1)
            color = 'tab:red'
            # Flux plot XX pol
            ax1.set_xlabel('Frames (2.56 us)')
            ax1.set_ylabel('Sys temp (XX pol)', color=color)
            ax1.plot(flux[0], color=color,alpha = 0.1)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.plot(np.nanmedian(flux[0]) + h[0],
                color=color,label = fr'$\sigma=${downsampling_factor}')
            ax1.axhline(np.nanmedian(flux[0]),color = color,linestyle = '--')
            ax1.axvline(pulse_lim[0], color = 'black')
            ax1.axvline(pulse_lim[1], color = 'black')
            ax1.legend(loc = 'lower left')
            #ax1.set_ylim(np.nanmedian(flux[0])  - 3 * rms[0],np.nanmax(flux[0]) + rms[0])
            ax1.set_ylim(np.nanmin(flux[0])  - rms[0],np.nanmax(flux[0]) + rms[0])
            # Inset XX pol
            axins1 = inset_axes(ax1, width = 1, height = 1, loc='upper left')
            axins1.plot(flux[0],color = color, alpha = 0.5)
            axins1.plot(h[0] + np.nanmedian(flux[0]),color = color)
            axins1.set_xlim(pulse_lim[0],pulse_lim[1]) # Limit the region for zoom
            axins1.set_ylim(np.nanmedian(flux[0]),np.nanmax(flux[0]))
            axins1.xaxis.set_ticklabels([])
            # Flux plot YY pol
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('Sys temp (YY pol)', color=color)  # we already handled the x-label with ax1
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.plot(flux[1], color=color,alpha = 0.1)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.plot(np.nanmedian(flux[1]) + h[1],
                color=color,label = fr'$\sigma=${downsampling_factor}')
            ax2.axhline(np.nanmedian(flux[1]),color = color,linestyle = '--')
            ax2.axvline(pulse_lim[0], color = 'black')
            ax2.axvline(pulse_lim[1]  , color = 'black')
            ax2.legend(loc = 'lower left')
            #ax2.set_ylim(np.nanmedian(flux[1]) - 3 * rms[1],np.nanmax(flux[1]) + rms[1])
            ax2.set_ylim(np.nanmin(flux[1]) - rms[1],np.nanmax(flux[1]) + rms[1])
            # Inset YY pol
            axins2 = inset_axes(ax2, width = 1, height = 1, loc='upper right')
            axins2.plot(flux[1],color = color, alpha = 0.5)
            axins2.plot(h[1]+ np.nanmedian(flux[1]),color = color)
            axins2.set_xlim(pulse_lim[0],pulse_lim[1]) # Limit the region for zoom
            axins2.set_ylim(np.nanmedian(flux[1]),np.nanmax(flux[1]))
            axins2.xaxis.set_ticklabels([])
            return ax1,ax2
        def plot_ds_waterfall(ax,wfall,pulse_lim,downsampling_factor):
            left = pulse_lim[0] - (pulse_lim[1] - pulse_lim[0])
            right = pulse_lim[1] + (pulse_lim[1] - pulse_lim[0])
            pww = _scrunch(np.abs(wfall[...,left:right])**2,tscrunch = downsampling_factor,fscrunch = 1)
            ax.axvline(pulse_lim[0],color = 'black')
            ax.axvline(pulse_lim[1],color = 'black')
            from matplotlib.colors import LogNorm
            ax.imshow(pww,
                      extent = [left,right,400,800],
                      aspect = 'auto',
                      norm = LogNorm(),
                      cmap = 'Greys')
            return ax
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2,2)
        ax_top   = fig.add_subplot(gs[0, :])
        ax_left  = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])
        plot_matched_filter_two_pols(ax_top,flux,h_offset,pulse_lim = lim,
                                    downsampling_factor = downsampling_factor)
        plot_ds_waterfall(ax_left,ww[:,0], pulse_lim = lim , 
                          downsampling_factor = downsampling_factor)
        plot_ds_waterfall(ax_right,ww[:,1], pulse_lim = lim, 
                          downsampling_factor = downsampling_factor)
        if type(diagnostic_plots) is str:
            plt.savefig(diagnostic_plots,dpi = 300)
            print(f'Saved plot to {diagnostic_plots}')
            plt.close('all')
        else:
            plt.show()
    if full_output:
        return data_clipped, h, ww, valid_channels,downsampling_factor,signal_temp,noise_temp, end_bin_frames, start_bin_frames, valid_time_range, w
    #I added stuff after noise_temp
    else:
        return 
    