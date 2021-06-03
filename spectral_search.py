# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:38:58 2021

@author: Eve
"""
import numpy as np
import matplotlib.pyplot as plt
from baseband_analysis.core.sampling import _upchannel as upchannel 
#this version allows me to upchannel a piece rather than everything, saving lots of time
from frb_spectral_search.inverse_macquart import inverse_macquart
import scipy.signal

#imports just for get_smooth_matched_filter_new
from matplotlib.pyplot import *
from scipy.stats import median_absolute_deviation
from baseband_analysis.analysis.snr import get_snr
from baseband_analysis.core.sampling import fill_waterfall, _scrunch, downsample_power_gaussian, clip
from baseband_analysis.core.signal import get_main_peak_lim, tiedbeam_baseband_to_power, get_weights
from common_utils import delay_across_the_band 

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def spectral_search_wrapper(data, DM=None, downsampling_factor=None, freq_id=None):
    #input: data = of type example( data = BBData.from_file('/data/frb-archiver/user-data/calvin/lcr_0.1.h5') )
    #runs whole process of selecting FRB time region, removing rfi, upchanneling, noise subtracting
    #and matched filtering to locate dips in the spectrum with error bars
    if freq_id is None:
        freq_id = data.index_map['freq']
    if DM is None:
        DM = data["tiedbeam_baseband"].attrs['DM']
    print("DM is " + str(DM))
    #have to auto set downsampling_factor
    if downsampling_factor == None:
        downsampling_factor = data["tiedbeam_power"].attrs["time_downsample_factor"]
    
    print(f'Using downsampling_factor_auto: {downsampling_factor}')
        
    
    data_clipped, h, ww, valid_channels,downsampling_factor,signal_temp,noise_temp, end_bin_frames, start_bin_frames, time_range, w = get_smooth_matched_filter_new(data, DM=DM, downsampling_factor = downsampling_factor)
    
    print("time_range")
    print(time_range)
    on_range = [start_bin_frames, end_bin_frames]
    n, noise_spec, noise_std = calculate_noise(ww, time_range, on_range, valid_channels)
    frbspec, frbindexes, spectra, spec_ebar  = plot_spectra(ww, start_bin_frames, end_bin_frames, valid_channels, time_range, DM, noise_spec, noise_std)
    upchaned = np.abs(frbspec[0,:,:])**2
    upchaned = np.sum(upchaned, 0)
    #width sizes to search over
    wids = [1,2.5,4]
    sig1 = -gaussian(np.linspace(-100,100,300), 0, wids[0])
    sig2 = -gaussian(np.linspace(-100,100,300), 0, wids[1])
    sig3 = -gaussian(np.linspace(-100,100,300), 0, wids[2])
    data1d, corrs = find_spectral_lines_multi(spectra, sig1, sig2, sig3, wids, spec_ebar)
    plt.show()
    return corrs, upchaned

def plot_spectra(ww, start_bin, end_bin, valid_channels, time_range, DM, noise_spec, noise_std):
    # Purpose: plotting the onburst spectrum with the Macquart region for expected 21 cm line off of DM
    #shows spectrum after processing (upchannelization and noise subtracting and derippling compared to original spectrum)
    #ww is the data rid of rfi and filled already
    my_freq_id = np.arange(1024)   
    frbspec, frbindexes, frbchan_id_upchan = upchannel(ww[:,:,start_bin:end_bin], my_freq_id)
    fluxup = np.sum(np.sum(np.abs(frbspec)**2,axis = 0),axis = 0)
    spectra = fluxup 
    #Correct for upchaannelization repeating 16 pattern
    fix = np.array([0.52225748, 0.58330915, 0.6868705, 0.80121821,
                         0.89386546, 0.95477358, 0.98662733, 0.99942558,
                         0.99988676, 0.98905127, 0.95874124, 0.90094667,
                         0.81113021, 0.6999944, 0.59367968, 0.52614263])
    big_fix = np.tile(fix, len(spectra)//16)
    spectra = np.multiply(spectra, 1/big_fix)
    spectra = spectra - noise_spec
    f_emit =1420.4 #MHz
    z = inverse_macquart(DM)
    a =  f_emit/(z+.3+1)
    b =  f_emit/(z-.3+1)
    fluxww = np.sum(np.sum(np.abs(ww[:,:,start_bin:end_bin])**2, axis = 1), axis= 1)
    #plotting
    plt.figure(11)
    plt.plot(frbindexes, np.repeat(fluxww, 16), color='#008a44', alpha = 1, label='Original Spectrum') #this is time vs flux we want frequency
        
        
    plt.errorbar(frbindexes, spectra, yerr = noise_std,ecolor= 'k', color='#006c8a', alpha= .5, label='Upchanneld Spectrum')
                
    plt.axvspan(a, b, alpha=0.2, color='orange', label='Expected 21 cm line based on DM')
    plt.legend()
    plt.show()
    return frbspec, frbindexes, spectra, noise_std

def calculate_noise(ww, time_range, on_range, valid_channels): 
    #Purpose: calculates noise average and standard deviation based off of off-pulse data
    #looks at chunks of time equal to the on-burst and processes them identically, 
    #ww has waterfall and rfi but this upchannelizes and deripples
    
    noise_ranges =[] 
    l_on = on_range[1]-on_range[0]
    bottom = on_range[0]-l_on

    while bottom > time_range[0]:
        noise_ranges.append([bottom, bottom+l_on])
        bottom -= l_on
    top = on_range[1]+l_on+1
    while top < time_range[1]-1:
        noise_ranges.append([top-l_on, top])
        top += l_on

    my_freq_id = np.linspace(0,1023, 1024) 

    #initialize list of spectrums
    x= noise_ranges[0]
    spec, frbindexes, frbchan_id_upchan = upchannel(ww[:,:,x[0]:x[1]], my_freq_id)
    spec1d = np.sum(np.sum(np.abs(spec)**2,axis = 0),axis = 0)
    fix = np.array([0.52225748, 0.58330915, 0.6868705, 0.80121821,
                          0.89386546, 0.95477358, 0.98662733, 0.99942558,
                          0.99988676, 0.98905127, 0.95874124, 0.90094667,
                          0.81113021, 0.6999944, 0.59367968, 0.52614263])
    big_fix = np.tile(fix, len(spec1d)//16)
    spec1d = np.multiply(spec1d, 1/big_fix)
    n_range_specs = spec1d
    count = 1
    for x in noise_ranges[1:]:      
        print("spec" + str(count) +" out of " + str(len(noise_ranges)))
        count +=1
        spec, frbindexes, frbchan_id_upchan = upchannel(ww[:,:,x[0]:x[1]], my_freq_id)
        spec1d = np.sum(np.sum(np.abs(spec)**2,axis = 0),axis = 0)
        spec1d = np.multiply(spec1d, 1/big_fix)
        added = np.append(n_range_specs, spec1d)
        n_range_specs = np.reshape(added, [-1, 1024*16]) #use -1 so it fill in that dimension

    return n_range_specs, np.nanmean(n_range_specs, 0), np.nanstd(n_range_specs, 0)

def find_spectral_lines_multi(upchaned, sig1, sig2, sig3, wids, spec_ebar):
    # finds the correlation between a match filter, matched to expected spectral line dips and finds the peaks of the correlation
    #compares highest peaks with next highest peaks to help classify the peaks as real
    #input: orig = upchannelized flux vs frequency data
    corr1 = scipy.signal.correlate(upchaned, sig1, mode='valid', method='fft' ) #pretty much same thing as match filter, also fftconvolve
    corr2 = scipy.signal.correlate(upchaned, sig2, mode='valid', method='fft' ) 
    corr3 = scipy.signal.correlate(upchaned, sig3, mode='valid', method='fft' ) 
    #normalization to -1
    n1 = scipy.signal.correlate( np.ones(1000), sig1, mode='valid', method='fft' )
    ncorr1 = corr1/(-n1[0])
    n2 = scipy.signal.correlate( np.ones(1000), sig2, mode='valid', method='fft' )
    ncorr2 = corr2/(-n2[0])
    n3 = scipy.signal.correlate( np.ones(1000), sig3, mode='valid', method='fft' )
    ncorr3 = corr3/(-n3[0])
    #fig7, (ax1, ax2) = plt.subplots(2, 1)
    
    fig71 = plt.figure(9)
    gs1 = fig71.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.98, wspace=0.05)
    ax1 = fig71.add_subplot(gs1[:-1, :])
    ax2 = fig71.add_subplot(gs1[-1, -3])
    ax3 = fig71.add_subplot(gs1[-1, -2])
    ax4 = fig71.add_subplot(gs1[-1, -1])

    shift = int(len(sig1)/2) #try this as not an int
    xaxis = np.linspace(shift, 1024-shift,len(upchaned))

    xaxis = np.linspace(800,400, 1024*16,endpoint=False)
    #ax1.errorbar(xaxis, upchaned, yerr = spec_ebar, alpha=.2, ecolor='k', label="spectrum")
    ax1.plot(xaxis, upchaned, color="#106bcc", label="spectrum")
    ax1.fill_between(xaxis, upchaned-spec_ebar, upchaned+spec_ebar, color = '#106bcc',alpha = .25)
    str1 = 'wid = ' + str(wids[0])
    str2 ='wid = ' + str(wids[1])
    str3 = 'wid = ' + str(wids[2])
    ax1.plot(xaxis[150:-149], ncorr1, color='#47c995',  alpha = .7, label=str1)
    ax1.plot(xaxis[150:-149], ncorr2, 'm',  alpha = .7, label=str2)
    ax1.plot(xaxis[150:-149], ncorr3, 'b',  alpha = .7, label=str3)
    good_peaks1, bad_peaks1 = find_good_matches(ncorr1, upchaned) 
    good_peaks2, bad_peaks2 = find_good_matches(ncorr2, upchaned)
    good_peaks3, bad_peaks3 = find_good_matches(ncorr3, upchaned)
    #ax1.set_ylim([0, 1e4])
    colors1 = ['#00e69d', '#00fc4c', '#a0fc00']
    colors2 = ['#ff00ff', '#d000ff', '#ff0062']
    colors3 = ['#2b00ff', '#0095ff', '#00eeff']
    for x in range(len(good_peaks1)):
        ax1.plot(xaxis[150:-149][good_peaks1[x]], (ncorr1[good_peaks1[x]]), marker = 'x', color = colors1[x], linestyle='')
    for x in range(len(good_peaks2)):
        ax1.plot(xaxis[150:-149][good_peaks2[x]], ncorr2[good_peaks2[x]], marker = 'x', color = colors2[x], linestyle='')
    for x in range(len(good_peaks3)):
        ax1.plot(xaxis[150:-149][good_peaks3[x]], ncorr3[good_peaks3[x]], marker = 'x', color = colors3[x], linestyle='')
    fig71.legend()
    
    wind = 25
    xaxis_peaks = xaxis[40-wind: 40+wind]-xaxis[40]
    for x in range(len(good_peaks1)):
        x1 = good_peaks1[x]   
        ax2.plot(xaxis_peaks, upchaned[x1+shift-wind: x1+shift+wind], color = colors1[x])
        ax2.fill_between(xaxis_peaks, upchaned[x1+shift-wind: x1+shift+wind]-spec_ebar[x1+shift-wind: x1+shift+wind], upchaned[x1+shift-wind: x1+shift+wind]+spec_ebar[x1+shift-wind: x1+shift+wind], color=colors1[x], alpha=.25)
    for x in range(len(good_peaks2)):
        x2 = good_peaks2[x]
        ax3.plot(xaxis_peaks, upchaned[x2+shift-wind: x2+shift+wind], color = colors2[x])
        ax3.fill_between(xaxis_peaks, upchaned[x2+shift-wind: x2+shift+wind]-spec_ebar[x2+shift-wind: x2+shift+wind], upchaned[x2+shift-wind: x2+shift+wind]+spec_ebar[x2+shift-wind: x2+shift+wind], color=colors2[x], alpha=.25)
    for x in range(len(good_peaks3)):
        x3 = good_peaks3[x]
        #olderror bar way
        #ax4.errorbar(xaxis_peaks, upchaned[x3+shift-wind: x3+shift+wind], yerr = spec_ebar[x3+shift-wind: x3+shift+wind], ecolor = 'k', color = 'b')
        ax4.errorbar(xaxis_peaks, upchaned[x3+shift-wind: x3+shift+wind], color = colors3[x])
        ax4.fill_between(xaxis_peaks, upchaned[x3+shift-wind: x3+shift+wind]-spec_ebar[x3+shift-wind: x3+shift+wind], upchaned[x3+shift-wind: x3+shift+wind]+spec_ebar[x3+shift-wind: x3+shift+wind], color=colors3[x], alpha=.25)
    ax3.set_xlabel("Frequency [MHz]")
    #all this just to create a common y label, from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
    ax = fig71.add_subplot(111)
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_facecolor('none') 
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_ylabel("Proportional to Flux")
    plt.show()
    return upchaned, [ncorr1, ncorr2, ncorr3]


def find_good_matches(corr, upchaned):
    #inputs: corr- cross-correlation of 1d-spectra and the signal (gaussian dip)
    #upchaned - the 1d-spectra
    #-outputs possible dips by finding maximum in corr and ignoring values close to zero in correlation and 1d-spectra 
    #since those are assumed to be rfi
    found_peaks= scipy.signal.find_peaks(corr)[0] #might consider putting a widths condition here that is determined by astrophysics
    indexpeak = (np.argsort(corr)[::-1]) #sorting to find the largest correlations and then will compare to peaks
    new_indexpeak = []
    for x,y in zip(indexpeak, upchaned[150:-149][indexpeak]):
        if x in found_peaks: #using two fold approach to not double count high parts of peak, could do this by specifying distance between points as well
            if y != 0 and corr[x] != 0:#removing rfi from consideration 
                new_indexpeak.append(x)
    good_prom= new_indexpeak[0:3]
    bad_prom=new_indexpeak[3:7]
    return good_prom, bad_prom
#incoherent_dedisp doesn't happen here index comes from that
#from Calvin, should exist and be able to be imported from somewhere but doesn't seem to be in analysis snr like I though
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
        return h
