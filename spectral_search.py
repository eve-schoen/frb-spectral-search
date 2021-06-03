# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:38:58 2021

@author: Eve
"""
import numpy as np
import maplotlib.pyplot as plt
from baseband_analysis.core.sampling import _upchannel as upchannel 
#this version allows me to upchannel a piece rather than everything, saving lots of time
from baseband_analysis.analysis import snr
from frb_spectral_search.inverse_macquart import inverse_macquart
import scipy.signal

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def spectral_search_wrapper_new(data, DM=None, downsampling_factor=None, freq_id=None):
    #input: data = of type example( data = BBData.from_file('/data/frb-archiver/user-data/calvin/lcr_0.1.h5') )
    #runs whole process of selecting FRB time region, removing rfi, upchanneling, noise subtracting
    #and matched filtering to locate dips in the spectrum with error bars
    if freq_id == None:
        freq_id = data.index_map['freq']
    if DM is None:
        DM = data["tiedbeam_baseband"].attrs['DM']
    print("DM is " + str(DM))
    #have to auto set downsampling_factor
    if downsampling_factor == None:
        downsampling_factor = data["tiedbeam_power"].attrs["time_downsample_factor"]
    
    print(f'Using downsampling_factor_auto: {downsampling_factor}')
        
    
    data_clipped, h, ww, valid_channels,downsampling_factor,signal_temp,noise_temp, end_bin_frames, start_bin_frames, time_range, w = snr.get_smooth_matched_filter(data, DM=DM, downsampling_factor = downsampling_factor)
    
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