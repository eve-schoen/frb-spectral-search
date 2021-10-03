from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def freq_fourth(xs, a):
    return a*xs**4
def exponential(xs, a, exp):
    return a*xs**exp
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def autocorr_naive_nan(x):
    N = len(x)
    return np.array([np.nanmean((x[iSh:]) * (x[:N-iSh])) for iSh in range(N)])
def cauchy_func(del_nu,  m, f_dc):
    return m / (del_nu**2 + f_dc**2)

def hmhw_cauchy(acf, exclude_zero = True): #hlaf max half width
    if exclude_zero:
        start = 1
        acf= acf[1:]
    else:
        start = 0
    xdata = np.linspace(start, len(acf), len(acf), endpoint=False)
    popt, pcov = curve_fit(cauchy_func, xdata, acf, bounds=([-np.inf, 0], [np.inf, np.inf]),   check_finite=False)
    print(popt)
    return popt[1]

def calc_fdcs(all_acf, bands):
    widths =[]
    for i in all_acf:
        if i != []:
            widths.append(abs(hmhw_cauchy(i, exclude_zero =True))*.024) #converting channels to MHz
        else:
            pass
        #widths.append(np.nan)
    plt.figure()
    bands = np.array(bands)
    xdatasim = np.linspace(bands[0], bands[-1], 100)
    ydata = np.array(widths)
    plt.plot(bands, (widths), '+')
    popt, pcov = curve_fit(freq_fourth,bands , ydata)
    plt.plot(xdatasim, (popt*xdatasim**4), label=r'fit='+str(popt)+ '(delta_nu)^4')
    plt.xlabel('Freq. [MHz]')
    plt.ylabel("f_dc")
    plt.legend()
    plt.show()

def acf_naive2(spectra, n, plot_range = None, plot=True):
    spec_bigsub = gauss_subtract_wo_and_replace_zero(spectra, 600)[150:-149] #double removing 150?
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

# def gauss_subtract_wo_and_replace_zero(spec , wid):
#     #returns spectrum minus the smoothed gaussian with zeros uneffected
#     bigsig = gaussian(np.linspace(-100,100,300), 0, wid)
#     sigarea = scipy.integrate.trapz(bigsig)
#     spec_no0 = spec[spec != 0]
#     smooth = scipy.signal.correlate(spec_no0, bigsig, mode='same', method='fft' )/sigarea
#     msub_spec = spec.copy()
#     msub_spec[spec != 0 ] = spec_no0/smooth #changed - to / just to try, no work,
#     return msub_spec[150:-149]
def gauss_subtract_wo_and_replace_zero(spec , wid):
    #returns spectrum minus the smoothed gaussian with zeros uneffected
    bigsig = gaussian(np.linspace(-100,100,300), 0, wid)
    sigarea = scipy.integrate.trapz(bigsig)
    spec_no0 = spec[spec != 0]
    bigcorr = scipy.signal.correlate(spec_no0, bigsig, mode='same', method='fft' )/sigarea
    msub_spec = spec.copy()
    msub_spec[spec != 0 ] = spec_no0-bigcorr #changed - to / just to try, no work
    return msub_spec[150:-149]


def acf_bands(spectra, n, plot_range = None, band_size = 40, normalize = True, skip = None):
#band size in MHz, calls naive acf calculation to perform ACF on individual frequency bands
    channel_sz = round(band_size/.024)
    num_bands = (1024*16//channel_sz)
    bands = []
    #fig, (ax1) = plt.subplots(111)
    colors = ['darkviolet', 'b','c', 'mediumseagreen', 'tab:olive', 'gold', 'tab:orange', 'r', 'lightcoral']
    fig, ax = plt.subplots(1, 1)
    start = 0 #counts channels of bands
    n2 = len(fig.axes)
    freqs = np.linspace(800, 400, 1024*16,endpoint=False)
    all_acf = []
    test = []
    for i in range(num_bands):
        if i in skip:
            all_acf.append([])
            pass
        else:
            #a
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
            ax.plot( acf+i*.2, alpha= 1,color = colors[i], label=lab)
            ax.plot(np.transpose(nacf) +i*.2, color = colors[i], alpha = .8/len(nacf))
            if i ==5:
                print('here')
                test.append(nacf)
            if plot_range != None:
                ax.set_xlim([0,plot_range])
                if plot_range < 101:
                    ax.xaxis.set_minor_locator(MultipleLocator(1))

        start += channel_sz
    ax.set_xlabel( "frequency lag (24 kHz channels)" )
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
    plt.show()
    return all_acf, bands, test

def plot_bands(all_acf, bands, remove_nan=True):
    if remove_nan:
        for i in range(len(all_acf)):
            b = np.array(all_acf[i])
            all_acf[i] = b[~np.isnan(b)]
    widths =[]
    for i in all_acf:

        if i != []:
            widths.append(abs(hmhw_cauchy(i, exclude_zero =True))*.024) #converting channels to MHz

    plt.figure()
    bands = np.array(bands)
    xdata = bands[:]
    ydata = np.array(widths)
    plt.plot(bands, (widths), '+')
    xs = np.linspace(xdata[0], xdata[-1], 100)
    popt1, pcov1 = curve_fit(freq_fourth,xdata , ydata)
    popt, pcov = curve_fit(exponential,xdata , ydata,  p0=[popt1[0], 4]) #initial guesses added to prevent runtime error
    plt.plot(xs, (popt[0]*xs**popt[1]), label=r'fit='+str(round(popt[0], 3))+' (delta_nu)^' +str(round(popt[1], 3)))
    plt.plot(xs, (popt1*xs**4),'r',  label=r'fit='+str(round(popt1[0], 3))+' (delta_nu)^4' )
    plt.xlabel('Freq. [MHz]', fontsize =16)
    plt.ylabel("f_dc [MHz]", fontsize =16)
    plt.legend()
def wrapper(noise_spec, spectra, n, noise_std, skip):
    a = noise_spec.copy()
    b= spectra.copy()
    c= n.copy()
    newn, newnoise, newspec = spectral_search_snr2.clean_rfi_more(c, a, b, noise_std)
    all_acf, bands= acf_bands(newspec, newn[0:30], plot_range = 100, band_size = 40, skip = skip)
    plot_bands(all_acf, bands)
def gauss_smooth_divide(spec , wid):
    #returns spectrum minus the smoothed gaussian with zeros uneffected
    bigsig = gaussian(np.linspace(-100,100,300), 0, wid)
    sigarea = scipy.integrate.trapz(bigsig)
    spec_no0 = spec[spec != 0]
    smooth = scipy.signal.correlate(spec_no0, bigsig, mode='same', method='fft' )/sigarea
    smoothdivided_spec = spec.copy()
    smoothdivided_spec[spec != 0 ] = spec_no0/smooth #changed - to / just to try, no work,
    smoothdivided_spec[spec == 0 ] = np.nan
    return np.array(smoothdivided_spec[150:-149])

def ACF_new(spec, n, plot_range = None, band_size = 40,  skip = []):
    smoothdivided_spec_on = gauss_smooth_divide(spec, 600)
    smoothdivided_spec_off = []
    for i in range(len(n)):
        smoothdivided_spec_off.append(gauss_smooth_divide(n[i], 600))
    channel_sz = round(band_size/.024)
    num_bands = (1024*16//channel_sz)
    bands = []
    colors = ['darkviolet', 'b','c', 'mediumseagreen', 'tab:olive', 'gold', 'tab:orange', 'r', 'lightcoral']
    fig, ax = plt.subplots(1, 1)
    start = 0
    n2 = len(fig.axes)
    freqs = np.linspace(800, 400, 1024*16,endpoint=False)
    all_acf = []
    set_scale=1
    for i in range(num_bands):
        if i in skip:
            all_acf.append([])
            pass
        else:
            acf_on = autocorr_naive_nan(smoothdivided_spec_on[start:start+channel_sz])

            acf_off = []
            for j in range(len(n)):
                acf_off_piece = autocorr_naive_nan(smoothdivided_spec_off[j][start:start+channel_sz])

                acf_off.append(acf_off_piece/acf_off_piece[0])
            print(len(acf_on))
            print(len(np.nanmean(acf_off, axis=0)))
            on_minus_off = acf_on - np.nanmean(acf_off, axis=0) #for each band
            final_acf = on_minus_off/on_minus_off[0] # divide by zero lag term

            all_acf.append(final_acf)

            lab =  str(round(freqs[start])) + '-'+ str(round(freqs[start+channel_sz])) + ' MHz'
            bands.append((freqs[start]+freqs[start+channel_sz])/2)
            ax.plot( final_acf+i*.2*set_scale, alpha= 1,color = colors[i], label=lab) #spacing out plots by .2 times general scale of first point
            print(i)
            ax.plot(np.transpose(acf_off) +i*.2*set_scale, color = colors[i], alpha = .8/len(acf_off))
            if plot_range != None:
                ax.set_xlim([0,plot_range])
                if plot_range < 101: #more fine grid lines is smaller plot range
                    ax.xaxis.set_minor_locator(MultipleLocator(1))

        start += channel_sz
    ax.set_xlabel( "frequency lag (24 kHz channels)" )
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
    plt.show()
    return all_acf, bands, acf_off

def acf_from_stratch(spec, n, plot_range = None, band_size = 40,  skip = []):
    #inputs band size in MHz
    band_size_channels = round(band_size/.024)
    smoothdivided_spec_on = gauss_smooth_divide(spec, 600)
    smoothdivided_spec_off = []
    colors = ['darkviolet', 'b','c', 'mediumseagreen', 'tab:olive', 'gold', 'tab:orange', 'r', 'coral', 'lightcoral']
    fig, ax = plt.subplots(1, 1)
    for i in range(len(n)):
        smoothdivided_spec_off.append(gauss_smooth_divide(n[i], 600))
    for i in range(len(spec)//band_size):
        spec_slice_on = smoothdivided_spec_on[i*band_size_channels:(i+1)*band_size_channels]
        acf_on = autocorr_naive_nan(spec_slice_on)
        acf_offs = []
        for j in range(len(n)):
            print(smoothdivided_spec_off.shape())
            piece = smoothdivided_spec_off[j][i*band_size_channels:(i+1)*band_size_channels]
            acf_offs.append(piece/piece[0])
        on_m_off = acf_on-np.nanmean(acf_offs)
        final_acf = on_m_off/on_m_off[0]
        ax.plot( final_acf+i*.2, alpha= 1,color = colors[i])
        ax.plot(np.transpose(acf_offs) +i*.2, color = colors[i], alpha = .8/len(acf_offs))
    return acf_offs
