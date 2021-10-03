# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:38:58 2021

@author: Eve
"""
#corr calculations for noise added and only searching corr far from rfi
import numpy as np
import matplotlib.pyplot as plt
from baseband_analysis.core.sampling import _upchannel as upchannel
#this version allows me to upchannel a piece rather than everything, saving lots of time
from frb_spectral_search.inverse_macquart import inverse_macquart
import scipy.signal
import scipy.integrate
import scipy.stats
from scipy.signal import lombscargle
from scipy.optimize import curve_fit
from astropy.convolution import convolve as astropy_convolve
import statsmodels.tsa.stattools
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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
    wids = [1,2.5,4]
    sigs = [-gaussian(np.linspace(-100,100,300), 0, wids[i]) for i in range(len(wids))]
    sig1 = -gaussian(np.linspace(-100,100,300), 0, wids[0])
    sig2 = -gaussian(np.linspace(-100,100,300), 0, wids[1])
    sig3 = -gaussian(np.linspace(-100,100,300), 0, wids[2])
    n, noise_spec, noise_std, nscorrs, nscorrs_stddev = calculate_noise(ww, time_range, on_range, valid_channels, sig1, sig2, sig3, sigs, scale=False)
    frbspec, frbindexes, spectra, spec_ebar  = plot_spectra(ww, start_bin_frames, end_bin_frames, valid_channels, time_range, DM, noise_spec, noise_std)
    upchaned = np.abs(frbspec[0,:,:])**2
    upchaned = np.sum(upchaned, 0)
    #width sizes to search over

    data1d, corrs = find_spectral_lines_multi(spectra, sig1, sig2, sig3, wids, spec_ebar, nscorrs_stddev, boxcorr)
    find_lines_SNR_space(spectra, corrs, spec_ebar, sigs, nscorrs_stddev, wids)
    plt.show()
    return corrs, upchaned
def find_lines_SNR_space(spectra, spec_ebar, sigs, noise_std, wids):
    snr = []
    bigsig = gaussian(np.linspace(-100,100,300), 0, 625)
    sigarea = scipy.integrate.trapz(bigsig)
    bigcorr = scipy.signal.correlate(spectra, bigsig, mode='same', method='fft' )/sigarea
    clip_spectra = spectra[150:-149]
    filt_spec=spectra-bigcorr
    for x,y in zip(spectra, range(len(filt_spec))):
        if x ==0:
            filt_spec[y] = 0  
    for i in range(len(sigs)):      
        sigarea = abs(scipy.integrate.trapz(sigs[i]))
        absorption = scipy.signal.correlate(filt_spec, sigs[i], mode='valid', method='fft' )/sigarea
        snr.append(absorption/noise_std[i][150:-149])
    peaks = []
    for i in range(len(sigs)):
        good_peaks, bad_peaks = find_good_matches(snr[i], filt_spec, wids[0]) #don't need bad peaks anymore...
        peaks.append(good_peaks)
    fig21 = plt.figure(21)
    gs1 = fig21.add_gridspec(nrows=4, ncols=3, left=0.05, right=0.98, wspace=0.05)
    ax1 = fig21.add_subplot(gs1[-3, :])
    ax6 = fig21.add_subplot(gs1[-2, :])
    ax2 = fig21.add_subplot(gs1[-1, -3])
    ax3 = fig21.add_subplot(gs1[-1, -2])
    ax4 = fig21.add_subplot(gs1[-1, -1])

    shift = int(len(sigs[0])/2) #try this as not an int
    xaxis = np.linspace(800,400, 1024*16,endpoint=False)
    
    ax1.plot(xaxis, filt_spec, color="#106bcc", label="filtered spectrum")
    ax1.fill_between(xaxis, filt_spec-spec_ebar, filt_spec+spec_ebar, color = '#106bcc',alpha = .25)
    strs = ['wid = ' + str(wids[i]) for i in range(len(wids))]
    colors = [['#00e69d', '#00fc4c', '#a0fc00'], ['#ff00ff', '#d000ff', '#ff0062'], ['#2b00ff', '#0095ff', '#00eeff']] 
    snrcolors= ['#47c995', 'm', 'b']
    #ax5= ax1.twinx()
    xaxis = xaxis[150:-149]
    for i in range(len(sigs)):
        ax6.plot(xaxis, snr[i], color=snrcolors[i],  alpha = .7, label=strs[i])
        colorcount = 0
        for x in range(len(peaks[i])): 
            ax6.plot(xaxis[peaks[i][x]], (snr[i][peaks[i][x]]), marker = 'x', color = colors[i][colorcount], linestyle='')
            ax1.plot(xaxis[peaks[i][x]], filt_spec[peaks[i][x]+150], marker = 'x', color = colors[i][colorcount], linestyle='')
            colorcount += 1
            if colorcount == len(colors[i]):
                colorcount = 0
                
    wind = round(8*wids[0]) #window goes out +/- 5 sigma  --> increasing to 8 to see more        
    xaxis_peaks = np.linspace(-wind*400/1024/16, wind*400/1024/16, 2*wind)[::-1]
    for x in range(len(peaks[0])):
        x1 = peaks[0][x]   
        ax2.plot(xaxis_peaks, filt_spec[x1+shift-wind: x1+shift+wind], color = colors[0][x])
        ax2.fill_between(xaxis_peaks, filt_spec[x1+shift-wind: x1+shift+wind]-spec_ebar[x1+shift-wind: x1+shift+wind], filt_spec[x1+shift-wind: x1+shift+wind]+spec_ebar[x1+shift-wind: x1+shift+wind], color=colors[0][x], alpha=.25)
    wind = round(5*wids[1]) 
    xaxis_peaks = np.linspace(-wind*400/1024/16, wind*400/1024/16, 2*wind)[::-1]
    for x in range(len(peaks[1])):
        x2 = peaks[1][x]
        ax3.plot(xaxis_peaks, filt_spec[x2+shift-wind: x2+shift+wind], color = colors[1][x])
        ax3.fill_between(xaxis_peaks, filt_spec[x2+shift-wind: x2+shift+wind]-spec_ebar[x2+shift-wind: x2+shift+wind], filt_spec[x2+shift-wind: x2+shift+wind]+spec_ebar[x2+shift-wind: x2+shift+wind], color=colors[1][x], alpha=.25)
    wind = round(5*wids[2]) 
    xaxis_peaks = np.linspace(-wind*400/1024/16, wind*400/1024/16, 2*wind)[::-1]
    for x in range(len(peaks[2])):
        x3 = peaks[2][x]
        ax4.plot(xaxis_peaks, filt_spec[x3+shift-wind: x3+shift+wind], color = colors[2][x])
        ax4.fill_between(xaxis_peaks, filt_spec[x3+shift-wind: x3+shift+wind]-spec_ebar[x3+shift-wind: x3+shift+wind], filt_spec[x3+shift-wind: x3+shift+wind]+spec_ebar[x3+shift-wind: x3+shift+wind], color=colors[2][x], alpha=.25)
    ax3.set_xlabel("Frequency [MHz]")
    #all this just to create a common y label, from https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
    ax = fig21.add_subplot(111)
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_facecolor('none')
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #ax.set_ylabel("Proportional to Flux")
    plt.show()
    fig21.legend()
    plt.tight_layout()
    plt.show()
    return snr, filt_spec

def plot_spectra(ww, start_bin, end_bin, valid_channels, time_range, DM, noise_spec, noise_std):
    # Purpose: plotting the onburst spectrum with the Macquart region for expected 21 cm line off of DM
    #shows spectrum after processing (upchannelization and noise subtracting and derippling compared to original spectrum)
    #ww is the data rid of rfi and filled already
    my_freq_id = np.arange(1024)
    frbspec, frbindexes, frbchan_id_upchan = upchannel(ww[:,:,start_bin:end_bin], my_freq_id)
    fluxup = np.sum(np.sum(np.abs(frbspec)**2,axis = 0),axis = 0)
    spectra = fluxup 
    #Correct for upchaannelization repeating 16 pattern
#     fix = np.array([0.52225748, 0.58330915, 0.6868705, 0.80121821,
#                          0.89386546, 0.95477358, 0.98662733, 0.99942558,
#                          0.99988676, 0.98905127, 0.95874124, 0.90094667,
#                          0.81113021, 0.6999944, 0.59367968, 0.52614263])

    fix = np.array([0.67461854, 0.72345924, 0.8339084 , 0.96487784, 1.0780604 ,
       1.1574216 , 1.2020986 , 1.2204636 , 1.2236487 , 1.2168874 ,
       1.1857561 , 1.1230892 , 1.0274547 , 0.9002419 , 0.7799086 ,
       0.68808776])
    big_fix = np.tile(fix, len(spectra)//16)
    spectra = np.multiply(spectra, 1/big_fix)
    spectra = spectra - noise_spec
    f_emit =1420.4 #MHz
    z = inverse_macquart(DM)
    a =  f_emit/(z+.3+1)
    b =  f_emit/(z-.3+1)
    fluxww = np.sum(np.sum(np.abs(ww[:,:,start_bin:end_bin])**2, axis = 1), axis= 1)
    fluxww2 = np.array([])
    for x in fluxww:
        for i in range(16):
            if x == 0:
                fluxww2 = np.append(fluxww2, np.nan)
            else:
                fluxww2 = np.append(fluxww2, x)
    #plotting
    plt.figure(11)



    plt.fill_between(frbindexes, spectra-noise_std,spectra+noise_std, color='#006c8a', alpha= .5)
    #plt.axvspan(a, b, alpha=0.2, color='orange', label='Expected 21 cm line based on DM')
    plt.plot(frbindexes, spectra, color='#006c8a', alpha= 1, label='Upchanneld Spectrum - Background')
    plt.plot(frbindexes,fluxww2 , color='#008a44', alpha = .8, label='Original Spectrum') #this is time vs flux we want frequency

    plt.xlabel('Freq. [MHz]', fontsize =16)
    plt.ylabel('Proportional to flux [arbitrary units]', fontsize =16)
    plt.legend()
    plt.show()
    return frbspec, frbindexes, spectra, noise_std

def calculate_noise(ww, time_range, on_range, valid_channels, sig1, sig2, sig3, sigs, scale =False):
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
#     fix = np.array([0.52225748, 0.58330915, 0.6868705, 0.80121821,
#                           0.89386546, 0.95477358, 0.98662733, 0.99942558,
#                           0.99988676, 0.98905127, 0.95874124, 0.90094667,
#                           0.81113021, 0.6999944, 0.59367968, 0.52614263])

###new new fix
    fix = np.array([0.67461854, 0.72345924, 0.8339084 , 0.96487784, 1.0780604 ,
       1.1574216 , 1.2020986 , 1.2204636 , 1.2236487 , 1.2168874 ,
       1.1857561 , 1.1230892 , 1.0274547 , 0.9002419 , 0.7799086 ,
       0.68808776])
    big_fix = np.tile(fix, len(spec1d)//16)
    spec1d = np.multiply(spec1d, 1/big_fix)
    n_range_specs = spec1d
    count = 1
    nscorr1 = nscorr2 = nscorr3 = []
    n1 = scipy.signal.correlate( np.ones(1000), sigs[0], mode='valid', method='fft' )[0]
    n2 = scipy.signal.correlate( np.ones(1000), sigs[1], mode='valid', method='fft' )[0]
    n3 = scipy.signal.correlate( np.ones(1000), sigs[2], mode='valid', method='fft' )[0]


    for x in noise_ranges[1:]:
        if count%5 ==0:
            print("spec" + str(count) +" out of " + str(len(noise_ranges)-1))
        count +=1
        spec, frbindexes, frbchan_id_upchan = upchannel(ww[:,:,x[0]:x[1]], my_freq_id)
        spec1d = np.sum(np.sum(np.abs(spec)**2,axis = 0),axis = 0)
        spec1d = np.multiply(spec1d, 1/big_fix)
        added = np.append(n_range_specs, spec1d)
        n_range_specs = np.reshape(added, [-1, 1024*16]) #use -1 so it fill in that dimension
        #should make a loop... do later
        speccorr1 = scipy.signal.correlate(spec1d, sig1, mode='valid', method='fft' )/abs(scipy.integrate.trapz(sig1))
        added = np.append(nscorr1, spec1d)
        nscorr1 = np.reshape(added, [-1, 1024*16]) #use -1 so it fill in that dimension
        speccorr2 = scipy.signal.correlate(spec1d, sig2, mode='valid', method='fft' )/abs(scipy.integrate.trapz(sig2))
        added = np.append(nscorr2, spec1d)
        nscorr2 = np.reshape(added, [-1, 1024*16]) #use -1 so it fill in that dimension
        speccorr3 = scipy.signal.correlate(spec1d, sig3, mode='valid', method='fft' )/abs(scipy.integrate.trapz(sig3))
        added = np.append(nscorr3, spec1d)
        nscorr3 = np.reshape(added, [-1, 1024*16])
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
    corr_ebar  = []
    for x in [nscorr1, nscorr2, nscorr3]:
        corr_ebar.append(scipy.stats.median_abs_deviation(x, axis=0, nan_policy = 'omit')*scale_fact)
    return n_range_specs, t_sys, spec_ebar, (nscorr1, nscorr2, nscorr3), corr_ebar, scale_fact_calc
def calculate_noise_simple(ww, time_range, on_range, valid_channels,scale =False):
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
    fix = np.array([0.67461854, 0.72345924, 0.8339084 , 0.96487784, 1.0780604 ,
       1.1574216 , 1.2020986 , 1.2204636 , 1.2236487 , 1.2168874 ,
       1.1857561 , 1.1230892 , 1.0274547 , 0.9002419 , 0.7799086 ,
       0.68808776])
    big_fix = np.tile(fix, len(spec1d)//16)
    spec1d = np.multiply(spec1d, 1/big_fix)
    n_range_specs = spec1d
    count = 1

    for x in noise_ranges[1:200]:
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
def find_spectral_lines_multi(upchaned, sig1, sig2, sig3, wids, spec_ebar, nscorrs_stddev):
    # finds the correlation between a match filter, matched to expected spectral line dips and finds the peaks of the correlation
    #compares highest peaks with next highest peaks to help classify the peaks as real
    #input: orig = upchannelized flux vs frequency data
    corr1 = scipy.signal.correlate(upchaned, sig1, mode='valid', method='fft' ) #pretty much same thing as match filter, also fftconvolve
    corr2 = scipy.signal.correlate(upchaned, sig2, mode='valid', method='fft' )
    corr3 = scipy.signal.correlate(upchaned, sig3, mode='valid', method='fft' )
    #normalization to -1
    n1 = scipy.signal.correlate( np.ones(1000), sig1, mode='valid', method='fft' )
    ncorr1 = corr1/abs(scipy.integrate.trapz(sig1))
    n2 = scipy.signal.correlate( np.ones(1000), sig2, mode='valid', method='fft' )
    ncorr2 = corr2/abs(scipy.integrate.trapz(sig2))#/(-n2[0])
    n3 = scipy.signal.correlate( np.ones(1000), sig3, mode='valid', method='fft' )
    ncorr3 = corr3/abs(scipy.integrate.trapz(sig3))#/(-n3[0])
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
    ax1.fill_between(xaxis[150:-149], ncorr1-nscorrs_stddev[0][150:-149], ncorr1+nscorrs_stddev[0][150:-149], color='#47c995',  alpha = .3)
    ax1.plot(xaxis[150:-149], ncorr2, 'm',  alpha = .7, label=str2)
    ax1.fill_between(xaxis[150:-149], ncorr2-nscorrs_stddev[1][150:-149], ncorr2+nscorrs_stddev[1][150:-149], color='m',  alpha = .3)
    ax1.plot(xaxis[150:-149], ncorr3, 'b',  alpha = .7, label=str3)
    ax1.fill_between(xaxis[150:-149], ncorr3-nscorrs_stddev[2][150:-149], ncorr3+nscorrs_stddev[2][150:-149], color='b',  alpha = .3)
    good_peaks1, bad_peaks1 = find_good_matches(ncorr1, upchaned, wids[0])
    good_peaks2, bad_peaks2 = find_good_matches(ncorr2, upchaned, wids[1])
    good_peaks3, bad_peaks3 = find_good_matches(ncorr3, upchaned, wids[2])
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


    for x in range(len(good_peaks1)):
        wind = round(5*wids[0]) #window goes out +/- 5 sigma
        xaxis_peaks = xaxis[40-wind: 40+wind]-xaxis[40]
        x1 = good_peaks1[x]
        ax2.plot(xaxis_peaks, upchaned[x1+shift-wind: x1+shift+wind], color = colors1[x])
        ax2.fill_between(xaxis_peaks, upchaned[x1+shift-wind: x1+shift+wind]-spec_ebar[x1+shift-wind: x1+shift+wind], upchaned[x1+shift-wind: x1+shift+wind]+spec_ebar[x1+shift-wind: x1+shift+wind], color=colors1[x], alpha=.25)
    for x in range(len(good_peaks2)):
        wind = round(5*wids[1])
        xaxis_peaks = xaxis[40-wind: 40+wind]-xaxis[40]
        x2 = good_peaks2[x]
        ax3.plot(xaxis_peaks, upchaned[x2+shift-wind: x2+shift+wind], color = colors2[x])
        ax3.fill_between(xaxis_peaks, upchaned[x2+shift-wind: x2+shift+wind]-spec_ebar[x2+shift-wind: x2+shift+wind], upchaned[x2+shift-wind: x2+shift+wind]+spec_ebar[x2+shift-wind: x2+shift+wind], color=colors2[x], alpha=.25)
    for x in range(len(good_peaks3)):
        wind = round(5*wids[2])
        xaxis_peaks = xaxis[40-wind: 40+wind]-xaxis[40]
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


def find_good_matches(corr, upchaned, wid):
    #inputs: corr- cross-correlation of 1d-spectra and the signal (gaussian dip)
    #upchaned - the 1d-spectra
    #-outputs possible dips by finding maximum in corr and ignoring values close to zero in correlation and 1d-spectra
    #since those are assumed to be rfi
    found_peaks= scipy.signal.find_peaks(corr)[0]#, width = 2*wid)[0] #might consider putting a widths condition here that is determined by astrophysics,

      #currently going for wid at least one sigma
    indexpeak = (np.argsort(corr)[::-1]) #sorting to find the largest correlations and then will compare to peaks
    new_indexpeak = []
    for x,y in zip(indexpeak, upchaned[150:-149][indexpeak]):
        if x in found_peaks: #using two fold approach to not double count high parts of peak, could do this by specifying distance between points as well
            #removing rfi and around rfi from consideration
            near = []
            for i in range(int(5*wid+1)):#don't want to be 8 sigma from rfi

                near.append(upchaned[150+(x-i)]==0)
                near.append(upchaned[150+(x+i)]==0)
            if np.all(np.logical_not(near)): #need all of near to be false
                new_indexpeak.append(x)
    good_prom= new_indexpeak[0:3]
    bad_prom=new_indexpeak[3:7]
    return good_prom, bad_prom
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

def autocorr(spectra, noise_spec, plot_range = None):
    noise_spec_bigsub = gauss_subtract_wo_and_replace_zero(noise_spec, 600)[150:-149]
    spec_bigsub = gauss_subtract_wo_and_replace_zero(spectra, 600)[150:-149]
    autocorr = scipy.signal.correlate(spec_bigsub, spec_bigsub, mode='full', method='fft')
    autocorr = autocorr[len(autocorr)//2:]
    nautocorr = scipy.signal.correlate(noise_spec_bigsub, noise_spec_bigsub, mode='full', method='fft')
    nautocorr = nautocorr[len(nautocorr)//2:]
    fig, (ax1) = plt.subplots(1, 1, sharex =True)
    ax5= ax1.twinx()
    ax5.plot(autocorr, 'r', alpha= .3, label='spectrum autocorrelation')
    ax1.plot(nautocorr, alpha = .7, label='noise autocorrelation')
    ax1.legend(loc='upper left')
    plt.legend()
    align_yaxis(ax1, ax5)
    if plot_range != None:
        plt.xlim([0,plot_range])
        if plot_range < 101:
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlabel( "frequency lag (24 kHz channels)" )
    plt.show()

def autocorr_norfi(spectra, noise_spec, plot_range = None):
    noise_spec_bigsub = gauss_subtract_wo_and_replace_zero(noise_spec, 600)[150:-149]
    spec_bigsub = gauss_subtract_wo_and_replace_zero(spectra, 600)[150:-149]
    noise_spec_bigsub = noise_spec_bigsub[noise_spec_bigsub!= 0]
    spec_bigsub = spec_bigsub[spec_bigsub !=0]
    autocorr = scipy.signal.correlate(spec_bigsub, spec_bigsub, mode='full', method='fft')
    autocorr = autocorr[len(autocorr)//2:]
    nautocorr = scipy.signal.correlate(noise_spec_bigsub, noise_spec_bigsub, mode='full', method='fft')
    nautocorr = nautocorr[len(nautocorr)//2:]
    fig, (ax1) = plt.subplots(1, 1, sharex =True)
    ax5= ax1.twinx()
    ax5.plot(autocorr, 'r', alpha= .3, label='spectrum autocorrelation')
    ax1.plot(nautocorr, alpha = .7, label='noise autocorrelation')
    ax1.legend(loc='upper left')
    plt.legend()
    align_yaxis(ax1, ax5)
    if plot_range != None:
        plt.xlim([0,plot_range])
        if plot_range < 101:
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlabel( "frequency lag (24 kHz channels)" )
    plt.show()
def acf_astropy(spectra, noise_spec, plot_range = None):
    #enter in spectra with nans in rfi and this should convolve without filling in missing data
    noise_spec_bigsub = gauss_subtract_wo_and_replace_zero(noise_spec, 600)[150:-149]
    spec_bigsub = gauss_subtract_wo_and_replace_zero(spectra, 600)[150:-149]
    noise_spec_bigsub = noise_spec_bigsub[noise_spec_bigsub!= 0]
    spec_bigsub = spec_bigsub[spec_bigsub !=0]
    #don't think preserve nan does quite what we want because it makes it nan again after convolution
    autocorr = astropy_convolve(spec_bigsub, spec_bigsub, mode='full',nan_treatment = 'interpolate', preserve_nan = True)
    autocorr = autocorr[len(autocorr)//2:]
    nautocorr = astropy_convolve(noise_spec_bigsub, noise_spec_bigsub, mode='full', method='fft')
    nautocorr = nautocorr[len(nautocorr)//2:]
    fig, (ax1) = plt.subplots(1, 1, sharex =True)
    ax5= ax1.twinx()
    ax5.plot(autocorr, 'r', alpha= .3, label='spectrum autocorrelation')
    ax1.plot(nautocorr, alpha = .7, label='noise autocorrelation')
    ax1.legend(loc='upper left')
    plt.legend()
    align_yaxis(ax1, ax5)
    if plot_range != None:
        plt.xlim([0,plot_range])
        if plot_range < 101:
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlabel( "frequency lag (24 kHz channels)" )
    plt.show()

def autocorr_naive_nan(x):
    N = len(x)
    return np.array([np.nanmean((x[iSh:]) * (x[:N-iSh])) for iSh in range(N)])
def autocorr_naive_plot(spectra, noise_spec, plot_range = None):
    noise_spec_bigsub = gauss_subtract_wo_and_replace_zero(noise_spec, 600)[150:-149]
    spec_bigsub = gauss_subtract_wo_and_replace_zero(spectra, 600)[150:-149]
    specnan = spec_bigsub.copy()
    specnan[noise_spec_bigsub ==0] = np.nan
    noisenan = noise_spec_bigsub.copy()
    noisenan[noise_spec_bigsub ==0] = np.nan
    autocorr = autocorr_naive_nan(specnan)
    nautocorr =  autocorr_naive_nan(noisenan)
    fig, (ax1) = plt.subplots(1, 1, sharex =True)
    ax5= ax1.twinx()
    ax5.plot(autocorr, 'r', alpha= .3, label='spectrum autocorrelation')
    ax1.plot(nautocorr, alpha = .7, label='noise autocorrelation')
    ax1.legend(loc='upper left')
    plt.legend()
    align_yaxis(ax1, ax5)
    if plot_range != None:
        plt.xlim([0,plot_range])
        if plot_range < 101:
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlabel( "frequency lag (24 kHz channels)" )
    plt.show()
def acf_statmodels(spectra, noise_spec, plot_range = None):
    noise_spec_bigsub = gauss_subtract_wo_and_replace_zero(noise_spec, 600)[150:-149]
    spec_bigsub = gauss_subtract_wo_and_replace_zero(spectra, 600)[150:-149]
    specnan = spec_bigsub.copy()
    specnan[noise_spec_bigsub ==0] = np.nan
    noisenan = noise_spec_bigsub.copy()
    noisenan[noise_spec_bigsub ==0] = np.nan
    autocorr = statsmodels.tsa.stattools.acf(specnan,nlags = 1024*16, adjusted = True, fft =False, missing = 'conservative')
    nautocorr =  statsmodels.tsa.stattools.acf(noisenan,nlags = 1024*16,adjusted = True,  fft =False, missing = 'conservative')
    fig, (ax1) = plt.subplots(1, 1, sharex =True)
    ax5= ax1.twinx()
    ax5.plot(autocorr, 'r', alpha= .3, label='spectrum autocorrelation')
    ax1.plot(nautocorr, alpha = .7, label='noise autocorrelation')
    ax1.legend(loc='upper left')
    plt.legend()
    align_yaxis(ax1, ax5)
    if plot_range != None:
        plt.xlim([0,plot_range])
        if plot_range < 101:
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlabel( "frequency lag (24 kHz channels)" )
    plt.show()
    return autocorr, nautocorr
def acf_statmodels2(spectra, n, plot_range = None):

    spec_bigsub = gauss_subtract_wo_and_replace_zero(spectra, 600)[150:-149]
    specnan = spec_bigsub.copy()
    specnan[spec_bigsub ==0] = np.nan


    autocorr = statsmodels.tsa.stattools.acf(specnan,nlags = 1024*16, adjusted = True, fft =False, missing = 'conservative')
    fig, (ax1) = plt.subplots(1, 1, sharex =True, sharey  = True)
    ax1.plot(autocorr, 'r', alpha= .3, label='spectrum autocorrelation')
    for i in range(len(n)):
        noise_spec_bigsub = gauss_subtract_wo_and_replace_zero(n[i], 600)[150:-149]
        noisenan = noise_spec_bigsub.copy()
        noisenan[noise_spec_bigsub ==0] = np.nan
        nautocorr =  statsmodels.tsa.stattools.acf(noisenan,nlags = 1024*16,adjusted = True,  fft =False, missing = 'conservative')

        if i == 1:
            ax1.plot(nautocorr, 'b', alpha = .05,label = 'noise autocorrelations')
        else:
            ax1.plot(nautocorr, 'b', alpha = .05)

    ax1.legend( loc='upper left')
    ax1.set_ylim([-1, 1.5])
            #align_yaxis(ax1, ax5)
    #ax1.legend(loc='upper left')
    plt.legend()

    if plot_range != None:
        plt.xlim([0,plot_range])
        if plot_range < 101:
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlabel( "frequency lag (24 kHz channels)" )
    plt.show()
    return autocorr, nautocorr
def acf_naive2(spectra, n, plot_range = None, plot=True):
    spec_bigsub = gauss_subtract_wo_and_replace_zero(spectra, 600)[150:-149]
    specnan = spec_bigsub.copy()
    specnan[spec_bigsub ==0] = np.nan
    autocorr = autocorr_naive_nan(specnan)
    if plot:
        fig, (ax1) = plt.subplots(1, 1, sharex =True)
        ax5= ax1.twinx()
    maxesum = 0
    allnautocorr = np.zeros([len(n), len(n[0])-598]) #loose 600 channels when gaussian subtracting....
    for i in range(len(n)):
        noise_spec_bigsub = gauss_subtract_wo_and_replace_zero(n[i], 600)[150:-149]
        noisenan = noise_spec_bigsub.copy()
        noisenan[noise_spec_bigsub ==0] = np.nan
        nautocorr =  autocorr_naive_nan(noisenan)
        maxesum += max(nautocorr)
        if i == 3:
             #using the last one doesn't procure good results and 1st is adjacent so this one
            if plot:
                ax5.plot(nautocorr, 'b', alpha = .8/len(n),label = 'noise autocorrelations')

        elif plot:
            ax5.plot(nautocorr, 'b', alpha = .8/len(n))
        allnautocorr[i, :] = nautocorr
    if plot:
        ax1.plot(autocorr, 'tab:pink', alpha= 1, label='spectrum autocorrelation')
    aunitn = maxesum/len(n)

    aunit = max(autocorr)
    if plot:
        ax1.set_ylim([-aunit/8, aunit+aunit/16])
        ax5.set_ylim([-aunitn/8, aunitn+aunitn/16])
        ax1.legend( loc='upper left')
        ax5.legend(loc='upper right')
        plt.legend()

        if plot_range != None:
            plt.xlim([0,plot_range])
            if plot_range < 101:
                ax1.xaxis.set_minor_locator(MultipleLocator(1))
        ax1.set_xlabel( "frequency lag (24 kHz channels)" )
        plt.show()
    return autocorr, allnautocorr

def acf_bands(spectra, n, plot_range = None, band_size = 40, normalize = True, skip = None):
#band size in MHz, calls naive acf calculation to perform ACF on individual frequency bands
    channel_sz = round(band_size/.024)
    num_bands = (1024*16//channel_sz)
    bands = []
    #fig, (ax1) = plt.subplots(111)
    colors = ['darkviolet', 'b','c', 'mediumseagreen', 'tab:olive', 'gold', 'tab:orange', 'r', 'lightcoral']
    fig, ax = plt.subplots(1, 1)
    start = 0
    n2 = len(fig.axes)
    freqs = np.linspace(800, 400, 1024*16)
    all_acf = []
    for i in range(num_bands):
        if i in skip:
            all_acf.append([])
            pass
        else:
            acf, nacf = acf_naive2(spectra[start:start+channel_sz], n[:, start:start+channel_sz], plot_range = plot_range, plot=False)
            all_acf.append(acf)
        #ax1.plot(np.transpose(nacf), 'b', alpha = .8/len(nacf),label = 'noise autocorrelations')
            lab =  str(round(freqs[start])) + '-'+ str(round(freqs[start+channel_sz])) + ' MHz'
            bands.append((freqs[start]+freqs[start+channel_sz])/2)
            if normalize:
                acf = acf/ (acf[0])
                maxes = nacf.max(axis=1)
                for j in range(len(nacf)):
                    nacf[j, :] =nacf[j, :]/maxes[j]
            ax.plot( acf+i*.2-.8, alpha= 1,color = colors[i], label=lab)
            ax.plot(np.transpose(nacf) +i*.2-.8, color = colors[i], alpha = .8/len(nacf))
            if plot_range != None:
                ax.set_xlim([0,plot_range])
                if plot_range < 101:
                    ax.xaxis.set_minor_locator(MultipleLocator(1))

        start += channel_sz
    ax.set_xlabel( "frequency lag (24 kHz channels)" , fontsize =16 )
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
    plt.ylabel('Correlation Amplitude [arbitrary units]', fontsize =16 )
    plt.show()
    return all_acf, bands

def acf_lombscarg(spectra, noise_spec, plot_range = None):
    noise_spec_bigsub = gauss_subtract_wo_and_replace_zero(noise_spec, 600)[150:-149]
    spec_bigsub = gauss_subtract_wo_and_replace_zero(spectra, 600)[150:-149]
    f = np.linspace(0.1, len(noise_spec)-.1, len(noise_spec))
    x = f[noise_spec != 0]
    y = spectra[noise_spec != 0]
    autocorr = lombscargle(x, y, f, normalize=False)
    yn = noise_spec[noise_spec != 0]
    nautocorr = lombscargle(x, yn, f, normalize=False)
    fig, (ax1) = plt.subplots(1, 1, sharex =True)
    ax5= ax1.twinx()
    ax5.plot(autocorr, 'r', alpha= .3, label='spectrum autocorrelation')
    ax1.plot(nautocorr, alpha = .7, label='noise autocorrelation')
    ax1.legend(loc='upper left')
    plt.legend()
    align_yaxis(ax1, ax5)
    if plot_range != None:
        plt.xlim([0,plot_range])
        if plot_range < 101:
            ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xlabel( "frequency lag (24 kHz channels)", fontsize =16 )
    plt.show()

def gauss_subtract_wo_and_replace_zero(spec , wid):
    #returns spectrum minus the smoothed gaussian with zeros uneffected
    bigsig = gaussian(np.linspace(-100,100,300), 0, wid)
    sigarea = scipy.integrate.trapz(bigsig)
    spec_no0 = spec[spec != 0]
    bigcorr = scipy.signal.correlate(spec_no0, bigsig, mode='same', method='fft' )/sigarea
    msub_spec = spec.copy()
    msub_spec[spec != 0 ] = spec_no0-bigcorr #changed - to / just to try, no work,
    #msub_spec[spec != 0 ] =msub_spec[spec != 0 ]  - np.median(msub_spec[spec != 0 ])#minus median
#     plt.figure()
#     plt.plot(msub_spec[150:-149])
#     print(np.median(msub_spec[spec != 0 ]))
    return msub_spec[150:-149]

def clean_rfi_more(n, noise_spec, spectra, noise_std):
    #alters origianl which I dont want :/
    noise_spec2 = np.copy(noise_spec)
    spectra2 = np.copy(spectra)
    n2 = np.copy(n)
    baseline  = np.median(noise_spec)
    new_rfiup = [noise_spec2> 3*noise_std+baseline]
    new_rfidown = [noise_spec2 < baseline - 3*noise_std]
    new_rfi = np.logical_or(new_rfiup, new_rfidown)[0, :]
    noise_spec2[new_rfi] = 0
    spectra2[new_rfi] = 0
    n2[:, new_rfi] = 0
    return n2, noise_spec2, spectra2

def half_max_width(acf):
    height = acf[0]
    val = height
    index = 0
    while val > height/2:
        index +=1
        val = acf[index]
    #returns first index that is less than height
    diffs = abs([acf[index-1], acf[index]]-height/2) #choose whether the one before or after crossing the threshold is closer
    if max(diffs) == diffs[0]:
        return index
    else:
        return index-1
def hmhw_cauchy(acf, exclude_zero = True): #hlaf max half width
    if exclude_zero:
        start = 1
        acf= acf[1:]
    else:
        start = 0
    xdata = np.linspace(start, len(acf), len(acf), endpoint=False)
    popt, pcov = curve_fit(cauchy_func, xdata, acf, bounds=([-np.inf, 0], [np.inf, np.inf]))
    print(popt)
    return popt[1]

def cauchy_func(del_nu,  m, f_dc):
    return m / (del_nu**2 + f_dc**2)
def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])
