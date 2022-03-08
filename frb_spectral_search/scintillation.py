from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.interpolate import make_lsq_spline
from frb_spectral_search.scint_and_spectral import clean_rfi_more
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def freq_fourth(xs, a):
    return a*xs**4
def exponential(xs, a, exp):
    return a*xs**exp
def autocorr_naive_nan(x):
    N = len(x)
    return np.array([np.nanmean((x[iSh:]) * (x[:N-iSh])) for iSh in range(N)])
def cauchy_func(del_nu,  m, f_dc):
    return m / (del_nu**2 + f_dc**2)

def hmhw_cauchy(acf, exclude_zero = True): #half max half width
    if exclude_zero:
        start = 1
        acf= acf[1:]
    else:
        start = 0
    xdata = np.linspace(start, len(acf), len(acf), endpoint=False)
    popt, pcov = curve_fit(cauchy_func, xdata, acf, bounds=([-np.inf, 0], [np.inf, np.inf]),   check_finite=False)
    print(popt)
    return popt[1]
        
def plot_bands(all_acf, bands,widths, error_bars,  remove_nan=True):
    if remove_nan:
        for i in range(len(all_acf)):
            b = np.array(all_acf[i])
            all_acf[i] = b[~np.isnan(b)]

    plt.figure()
    bands = np.array(bands)
    xdata = bands[:]
    widths = np.array(widths)
    ydata = widths
    error_bars = np.array(error_bars)
    large = np.array(error_bars > 10)
    plt.errorbar(bands[~large], (widths[~large]), yerr = error_bars[~large],fmt= 'o')
    xs = np.linspace(xdata[0], xdata[-1], 100)
    popt1, pcov1 = curve_fit(freq_fourth,xdata , ydata, sigma = error_bars, absolute_sigma = False)
    
    popt, pcov = curve_fit(exponential,xdata , ydata,   sigma = error_bars, absolute_sigma = False) #initial guesses added to prevent runtime error
    #p0=[popt1[0], 4],
    perr1 = np.sqrt(np.diag(pcov1))#*(600)**4 #one standard deviation error
    perr = np.sqrt(np.diag(pcov))[0]
    experr = np.sqrt(np.diag(pcov))[1]
    plt.plot(xs, (popt[0]*(xs/600 )**popt[1]), label=r'fit='+str(round(popt[0], 3))+' (delta_nu)^' +str(round(popt[1], 3)) + '+/-'+ str(round(experr, 2)))
    plt.plot(xs, (popt1*(xs/600)**4),'r',  label=r'fit='+str(round(popt1[0], 3))+' (delta_nu)^4' )
    plt.xlabel('Freq. [MHz]', fontsize =16)
    plt.ylabel(r"$\nu_{dc}$ [MHz]", fontsize =16)
    plt.legend()
    fdc600 = str(popt1) + '+/-' +  str(perr1) 
    
    fdc600_powerlaw = str(popt[0]) + '+/-' +  str(perr)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    print('freq^4.4 scaling')
    print(f"{popt1} +/- {perr1}")
    print(fdc600)
    print('freq^exp scaling')
    print(f"{popt[0]} +/- {perr}")
    print(fdc600_powerlaw)
    return popt1

def wrapper(noise_spec, spectra, n, noise_std, skip):
    a = noise_spec.copy()
    b= spectra.copy()
    c= n.copy()
    newn, newnoise, newspec = spectral_search_snr2.clean_rfi_more(c, a, b, noise_std)
    all_acf, bands= acf_bands(newspec, newn[0:30], plot_range = 100, band_size = 40, skip = skip)
    plot_bands(all_acf, bands)

def spline_divide_old(spec_on, spec_offs):
    #returns spectrum (on and off ones) minus the smoothed spline of the spectrum with zeros uneffected
    n = spec_offs.copy() #to avoid rewriting spec_offs
    spec_on_no0 = spec_on[spec_on != 0]
    xs  = np.arange(len(spec_on_no0))
    #setting knots of spline, must satisfy Schoenberg-Whitney conditions and this is doing that
    #from scipy make_lsq_spline documentation and numbers were set experimentally to match a gaussian filter around 15 MHz
    t = np.linspace(xs[1], xs[-2], num= 50) 
    k = 3
    t = np.r_[(xs[0],)*(k+1),
          t,
          (xs[-1],)*(k+1)]
    #
    spline = make_lsq_spline(xs, spec_on_no0, t, k= 3) 
    smooth = spline(xs)
    smoothdivided_spec = spec_on.copy()
    smoothdivided_spec[spec_on != 0 ] = spec_on_no0/smooth #changed - to / just to try, no work,
    smoothdivided_spec_offs  = []
    for i in range(len(n)):
        temp = n[i]
        #dividing by smooth on spectrum creates acf in off spectrum so.. creating smooth for each
        
        spline = make_lsq_spline(xs, temp[temp != 0 ], t, k= 3) 
        temp[temp != 0 ] = temp[temp!= 0 ]/spline(xs)
        temp[temp==0] = np.nan
        smoothdivided_spec_offs.append(temp)
    smoothdivided_spec[spec_on == 0 ] = np.nan
    return np.array(smoothdivided_spec) , np.array(smoothdivided_spec_offs)
def spline_divide(spec_on, spec_offs, num_splines = None):
    #returns spectrum minus the smoothed gaussian with zeros uneffected
    n = spec_offs.copy() #to avoid rewriting spec_offs
    spec_on_no0 = spec_on[spec_on != 0]
    xs  = np.arange(len(spec_on_no0))
    if num_splines is None:
        num_splines = 50
    t = np.linspace(xs[1], xs[-2], num= num_splines)
    k = 3
    t = np.r_[(xs[0],)*(k+1),
          t,
          (xs[-1],)*(k+1)]
    spline = make_lsq_spline(xs, spec_on_no0, t, k= 3) 
    smooth = spline(xs)
    smoothdivided_spec = spec_on.copy()
    smoothdivided_spec[spec_on != 0 ] = spec_on_no0/smooth #changed - to / just to try, no work,
    smoothdivided_spec_offs  = []
    
    for i in range(len(n)):
        temp = n[i]
        #dividing by smooth on spectrum creates acf in off spectrum so.. creating smooth for each
        if len(temp[temp != 0 ]) == 0 : #one of the noise_specs is all zeros... issue
        #issue for <50 noise sequences becasue temp becomes zeros
            temp = np.zeros([len(xs)])
            spline = make_lsq_spline(xs, temp, t, k= 3) 
        else:
            spline = make_lsq_spline(xs, temp[temp != 0 ], t, k= 3) 
            temp[temp != 0 ] = temp[temp!= 0 ]/spline(xs)
        temp[temp==0] = np.nan
        smoothdivided_spec_offs.append(temp)
    smoothdivided_spec[spec_on == 0 ] = np.nan
    return np.array(smoothdivided_spec) , np.array(smoothdivided_spec_offs), smooth
    
    
def acf_normalized(spec, n, plot_range = None, band_size = 40,  skip = [], event_id =None, num_splines= None):
    #inputs band size in MHz
    freqs = np.linspace(800, 400, num=1024*16, endpoint =False)
    band_size_channels = round(band_size/.02439024)
    smoothdivided_spec_on, smoothdivided_spec_off,smooth = spline_divide(spec, n)#, num_splines) need to go get updated spline divide later
    colors = ['darkviolet', 'b','c', 'mediumseagreen', 'tab:olive', 'gold', 'tab:orange', 'r',  'lightcoral' ]
    fig, ax = plt.subplots(1, 1)
    all_acf = []
    bands = []
    start = 0
    num_bands = len(spec)//band_size_channels
    widths = []
    perrs = []
    offs_std = []
    for i in range(num_bands):
        if i in skip:
            all_acf.append([])
            offs_std.append([])
            pass
        else:
            spec_slice_on = smoothdivided_spec_on[i*band_size_channels:(i+1)*band_size_channels]
            acf_on = autocorr_naive_nan(spec_slice_on-1)
            acf_offs = np.empty(smoothdivided_spec_off[:,i*band_size_channels:(i+1)*band_size_channels].shape)
            for j in range(len(n)):
                piece = smoothdivided_spec_off[j]
                piece = np.array(piece[i*band_size_channels:(i+1)*band_size_channels])
                acf_piece = autocorr_naive_nan(piece-1) #normalizating from one to zero...
                acf_offs[j] = (acf_piece/acf_piece[0]) #normalizing by the point at zero because that is biased by random noise
            final_acf = acf_on#/acf_on[0]
            std = np.std(acf_offs, axis = 0) #want size to match on and don't want to include rfi channels so not using np.nanstd 
            offs_std.append(std[~np.isnan(std)])
            all_acf.append(final_acf)
            print(i)
            scale= 1
            lab =  str(round(freqs[start+band_size_channels])) + '-'+ str(round(freqs[start])) + ' MHz'
            bands.append((freqs[start]+freqs[start+band_size_channels])/2)
            
            xs1 = np.arange(len(final_acf))*.024
            ax.plot(xs1,  final_acf+(num_bands-len(skip)-i+1)*.2*scale-.2, alpha= 1,color = colors[i], label=lab)
            ax.plot(xs1, np.transpose(acf_offs) +(num_bands-len(skip)-i+1)*.2*scale-.2, color = colors[i], alpha = .8/len(acf_offs))
                      
            if plot_range != None:
                ax.set_xlim([0,plot_range*.024])
                if plot_range < 101: #more fine grid lines is smaller plot range
                    ax.xaxis.set_minor_locator(MultipleLocator(1))
        start += band_size_channels
        
    #fitting cauchy function to each acf abnd
    widths =[]
    fdc_error_bars = []
    count = 0
    if plot_range != None:
        truncate = plot_range
    else:
        truncate = 0
    for i in all_acf: 
        if i != []:
            b=np.array(i)
            b = b[~np.isnan(b)]
            try:
                popt, pcov  = hmhw_cauchy(b, sigma = offs_std[count]  , exclude_zero =True, truncate = truncate)
                perr = np.sqrt(np.diag(pcov)) #one standard deviation error
                print('m: ' + str(popt[0]) + '+/-'+str(pcov[0]))
                widths.append(abs(popt[1])*.02439024) #converting channels to MHz
                fdc_error_bars.append(perr[1] *.02439024)
                if abs(popt[0]) > 3*abs(perr[0]): # only want signifgant points
                    xs  = (np.arange(len(final_acf))+1)
                
            #sometimes it cannot fit this well
                    ax.plot(xs1+.024, cauchy_func(xs, popt[0], popt[1])+(num_bands-len(skip)-count+1)*.2*scale-.2, 'k--', alpha = .8 )
   
                    
            except:
                print('not able to fit to find bandwidth for ' +str(count)+ " band")
        
        count += 1
    ax.set_ylim([-0.5, 3])
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    ax.set_xlabel( r"$\Delta \nu$ (MHz)" , fontsize =16)
    ax.set_ylabel(r"$\mathbf{r}(\Delta \nu)$", fontsize =16)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0., prop={'size': 13})
    if event_id is not None:
        plt.tight_layout()
        plt.savefig('plots/'+str(event_id)+'acf_bandsize'+str(band_size)+'plot_range'+str(plot_range), dpi=300)
    return all_acf, bands, widths, fdc_error_bars, smooth
def run_acf(spectras, noise_spec, n, noise_std,plot_range = 100, band_size = 40, skip = [], i=None, event_id=None):
    #i indicates polarization if none assume combined
    if i is None:
        newn =  np.nansum(n, axis=0)
        newspec = np.nansum(spectras, axis=0)
        pol='both'
    else:
        newn=n[i,:,:]
        newspec=  spectras[i,:]
        pol=str(i)
    print(newn.shape)
    all_acf, bands, widths, error_bars, smooth = acf_normalized(newspec, newn, plot_range = plot_range, band_size = band_size, skip = skip)
    if event_id is not None:
        plt.savefig('/data/user-data/eveschn/code/plots/'+str(event_id)+'acf_bandsize'+str(band_size)+'plot_range'+str(plot_range)+'pol'+pol, dpi=300)
    return all_acf, bands, widths, error_bars, smooth

def save_data(data, event_id, spectra, n, smooth=None):
    try:
        del data['tiedbeam_baseband'] 
    except:
        print('no tiedbeam_baseband found to delete, continuing saving')
    #data['fine_spectrum'] = spectra
    n_off_pulses = 50
    nfreq=1024
    npol=2
    data.create_dataset('fine_spectrum' , data = spectra, shape = (npol,nfreq * 16))
    
    d = n.shape
    if d[1] >= n_off_pulses:
        fine_spectrum_offs = n[:,:n_off_pulses,:]
    else:
        n_off_pulses = d[1]
        fine_spectrum_offs = n
        print(n.shape)
        print('Warning ' + str(d[0]) + ' off pulses, less than 50')
    data.create_dataset('fine_spectrum_offs',data = n, shape = (npol,n_off_pulses, nfreq * 16))
    if smooth is not None:
        data.create_dataset('smooth_spectrum',data = smooth) 
    #no shape for now because shape is smaller than fine spectra without zeros because edge affects
    
    

#     data.create_dataset('fine_spectrum',shape = (npol, nfreq * 16))
#     data['fine_spectrum'][:] = spectra
#     (i, n_off_pulses, j ) = n.shape
#     data.create_dataset('fine_spectrum_offs',maxshape = ( npol, n_off_pulses,nfreq * 16)) 
#     d = n.shape
#     if d[-1] > 50:
#         fine_spectrum_offs = n[:,:50, :]
#     data['fine_spectrum_offs'][:]=fine_spectrum_offs
#     data.create_dataset('smooth_spectrum',shape = (npol, nfreq * 16))
#     data['smooth_spectrum'][:] = smooth
    different_filename = '/data/user-data/eveschn/code/data_products/' +  str(event_id) + 'scint_processed'                    
    data.save(different_filename)