# coding: utf-8

#import matplotlib
#matplotlib.use('Qt4Agg')


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import numpy as np
#from astropy.io import fits
#from sklearn import datasets
#from sklearn.decomposition import FastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from scipy import linalg
from sklearn.preprocessing import StandardScaler


def user_rc(lw=1.5, fontsize=10, figsize=(8, 5)):
    """Set plotting RC parameters to make plots more readable"""
    plt.rc('lines', linewidth=lw)
    plt.rc('axes', lw=1, labelsize=18, titlesize=22)
    plt.rc('font', size=14, weight='normal')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=14)
    plt.rc('xtick.major', size=6, width=1)
#    plt.rc('axes', labelsize=18, titlesize=22)
    plt.rc('figure', titlesize=22, figsize=figsize)
    return
    
user_rc()

def get_Nbandpass(lam_i, lam_f, bp=0.1):
    """Returns number of bandpasses given initial & final wavelength, 
    and bandpass/fractional width"""
    return int(np.ceil(np.log10(lam_f/lam_i) / np.log10(1.+bp)))+1

def get_effective_bp(lam_i, lam_f, Nbp):
    """Returns the bandpass values given initial & final wavelengths, 
    and number of bands"""
    return (lam_f/lam_i)**(1./Nbp) - 1.

def get_central_bandpass(lam_i, lam_f, bp=0.1, cbp=0.1, Nbp=None, ret_all=False):
    """Returns central, left- and right-edge wavelengths of bandpasses"""
    if Nbp is None:
        Nbp = get_Nbandpass(lam_i, lam_f, bp=bp)
    print("Number of bandpasses = {}".format(Nbp))
    lam_central = np.clip(np.array([lam_i*(1.+cbp)**n * (2. + cbp) / 2. \
    						for n in range(Nbp)]), lam_i, lam_f)
    if ret_all:
        return lam_central, lam_central*(1-bp/2.), lam_central*(1.+bp/2.)
    return lam_central

def get_avg_albedo(x, y, bin_left, bin_right):
    """Computes the mean albedo (y val) given wavelength (x val) bins"""
    results = np.array([np.nanmean(y[(x>=bin_left[ii]) * (x<bin_right[ii])]) \
    						for ii in range(len(bin_left))])
    return results

def get_colour_inds(Nbp, Nchoose=2):
    """Returns Nchoose-wise indices corresponding to all unique 
    Nchoose-wise combinations of bandpasses (e.g., indices for colour). 
    For example, given 3 base bandpasses {b0 b1 b2}, Nchoose=2 yields 
    [[b0, b1], [b0, b2], [b1, b2]]. """
    return list(itertools.combinations(range(Nbp), Nchoose))

def get_colour_matrix(x, Nbp):
    """Returns colours (difference in magnitudes), given array of 
    bandpass magnitudes (magnitude = log10(flux))"""
    band_inds = get_colour_inds(Nbp)
    return np.array([x[i]/x[j] for i,j in band_inds])

def n_choose_2(n):
    """Returns number of combinations for N choose 2"""
    return n*(n-1)/2.

def make_spectrum_uniform(x, x0, y0):
    """Transform albedo spectrum on the same wavelength grid, where x is new 
    grid and x0, y0 is base values"""
    return np.interp(x, x0, y0)

def get_radius(mass, logg):
    """Computes effective radius of exoplanet given mass and logg"""
    radius = np.sqrt(mass/(10**logg/981.))
    return radius

def simulate_planets(loggs, mi=1.0, mf=10., Nm=10, logspace=True):
    """Simulate mass and radius of planets given array of logg (density), 
    number of mass bins and initial & final mass"""
    if logspace:
        mass = 10**np.linspace(np.log10(mi), np.log10(mf), Nm)
    else:
        mass = np.linspace(mi, mf, Nm)
    radius = np.zeros((mass.shape[0], loggs.shape[0]))
    for ii in range(mass.shape[0]):
        radius[ii,:] = get_radius(mass[ii], loggs)
    return mass, radius

def get_sub_neptune_radius(Mp, f_env=0.01, Fp=1.0, age=5.):
    """Approx M-R relation from Lopez & Fortney (2014)
    https://iopscience.iop.org/article/10.1088/0004-637X/792/1/1/"""
    Renv = 2.06 * (Mp**-0.21) * ((f_env/5.)**0.59) * \
            (Fp**0.044) * ((age/5.)**-0.18)
    Rcore = Mp**0.25
    return Renv + Rcore

def load_mn_spectra(mrange, grange, Trange, fsedrange, Nwavelength=2000,
                    prefix='data/Reflection_Spectra_Repository/'):
    """Loads in mini/sub-neptune spectra from MacDonald+2018, 
    with size = (mrange, grange, Trange, fsedrange, Nwavelength, 2)"""
#    mrange = np.array([0, 0.5, 1.0, 1.5, 2.0])
#    grange = np.arange(2.0, 3.7, 0.2)
#    Trange = np.arange(150, 401, 20).astype(int)
#    fsedrange = np.arange(10).astype(int)[::2]+1

    data = np.zeros((mrange.shape[0], grange.shape[0], 
                     Trange.shape[0], fsedrange.shape[0], Nwavelength, 2))
    for i,j,k,l in itertools.product(range(mrange.shape[0]), 
                                     range(grange.shape[0]),
                                     range(Trange.shape[0]),
                                     range(fsedrange.shape[0])):
        bad=np.zeros(Nwavelength).astype(bool)
        bad[572]=True #known bad index from correspondence w/MacDonald+2018
        fname = prefix+'m{0}/g{1}/T{2}/Albedo_spectra_m{0}_g{1}_T{2}_f{3}.dat'.format(
                                mrange[i], grange[j], Trange[k], fsedrange[l])
        data[i,j,k,l,:,:] = np.loadtxt(fname, skiprows=1, usecols=(0,1))
        bad = bad | (data[i,j,k,l,:,1]>1.)
        data[i,j,k,l,bad,1] = np.interp(data[i,j,k,l,bad,0], data[i,j,k,l,:,0][~bad], 
                                        data[i,j,k,l,:,1][~bad])
    return data

def load_ss_spectra(indir='data/ga_ss_spectra/'):
    """Load in SS object spectra from VPL. Hard coded"""
    modern_earth = np.loadtxt(indir+'Earth_geo_albedo.txt')
    hazy_archean_earth = np.loadtxt(indir+'Hazy_ArcheanEarth_geo_albedo.txt')
    jupiter = np.loadtxt(indir+'Jupiter_geo_albedo.txt')
    neptune = np.loadtxt(indir+'Neptune_geo_albedo.txt')
    proterozoic_hiO2 = np.loadtxt(indir+'proterozoic_hi_o2_geo_albedo.txt')
    proterozoic_loO2 = np.loadtxt(indir+'proterozoic_low_o2_geo_albedo.txt')
    uranus = np.loadtxt(indir+'Uranus_geo_albedo.txt')
    saturn = np.loadtxt(indir+'Saturn_geo_albedo.txt')
    return modern_earth, proterozoic_hiO2, proterozoic_loO2, hazy_archean_earth,\
            jupiter, neptune, uranus, saturn
#    ## warm nep. spectra from Kaltenegger (?) group, smilar to MacDonald+2018
#    warm_nep1au_cloud = np.loadtxt(indir+'warm_neptune_1au_clouds.txt')
#    warm_nep1au_nocloud = np.loadtxt(indir+'warm_neptune_1au_noclouds.txt')
#    warm_nep2au = np.loadtxt(indir+'warm_neptune_2au.txt')
#
#    # the warm nep spectra have different units
#    warm_nep1au_cloud[:,0]*=1e-3
#    warm_nep1au_nocloud[:,0]*=1e-3
#    warm_nep2au[:,0]*=1e-3
        
def get_noise(x, dx=0.11):
    """Simulate random white noise (Gaussian distribution with scale dx)"""
    return np.random.normal(0, scale=dx, size=x.shape)
    
    
def map_obs_to_fiducial(obs_ind, obs_labels, header, mrange, grange, 
                        Trange, fsedrange):
    """Map raveled index ("fake observations") to original 
    array[mrange,grange,Trange,fsedrange] simulation grid"""
    
    # compute g in fake observations using mass and radius
    g_ind = np.argmin(abs(np.log10(obs_labels[obs_ind,header['mass']] * \
                    1./obs_labels[obs_ind,header['radius']]**2 * 981)-grange))
    m_ind = np.argmin(abs(obs_labels[obs_ind,header['m']]-mrange))
    T_ind = np.argmin(abs(obs_labels[obs_ind,header['T']]-Trange))
    fsed_ind = np.argmin(abs(obs_labels[obs_ind,header['fsed']]-fsedrange))
    return m_ind, g_ind, T_ind, fsed_ind

def get_ith_test_set(test_set, test_set_weight, subset_size, Nsubsets=10, 
                     seed=True):
    """Generate random subsets (Nsubsets x subset_size) of the test set"""
    test_subset = np.zeros((subset_size, Nsubsets)).astype(int)
    for jj in range(Nsubsets):
        if seed:
            np.random.seed(jj)
        test_subset[:,jj] = np.random.choice(test_set, size=subset_size, 
                                  replace=False, p=test_set_weight)
    return test_subset


def plot_filter_set(lam_i, lam_f, lam_c, filter_choice=None, ax=None, ylabel=''):
    """Visualize the different combinations of filters"""
    Nbp = len(lam_c)
    alphas = np.ones(Nbp)*0.5
    if filter_choice is not None:
        alphas[filter_choice] = 0.9
        
    if ax is None:
        fig = plt.figure(figsize=(10,1))
        ax = fig.add_subplot(111)
        
    for cnter, lstuff in enumerate(zip(lam_c, lam_i, lam_f)):
        lc, li, lf = lstuff
        if cnter in filter_choice:
            p = patches.Rectangle((li,0), lf-li, 10, linewidth=0, 
                                  color='C{}'.format(cnter), alpha=0.8,
                             hatch='//', zorder=Nbp+1)
        else:
            p = patches.Rectangle((li,0), lf-li, 10, linewidth=0, 
                                  color='C{}'.format(cnter), 
                                  linestyle='--', alpha=0.7)
        ax.add_patch(p)
    ax.set_xlim(lam_i[0], lam_f[-1])
    ax.set_xticks(np.round(lam_c,0).astype(int))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='y', which='both',length=0)
    ax.set_yticks([])
    ax.set_ylabel(ylabel)
    return ax

def apply_LDA(Nchoose, features, labels, train_set, test_set, test_subsets, Nbp, 
              acc_thresh=None, topX=5, plot=True):
    """For each Nchoose bandpass/filter combinations, train LDA & then apply 
    model to Nsubset test sets to compute best bp/filter combo indices, and 
    the corresponding mean & std of accuracy."""
    Nsubsets = test_subsets.shape[-1]
    colour_inds = np.array(get_colour_inds(Nbp, Nchoose=Nchoose))
    accuracy = np.zeros((len(colour_inds), Nsubsets))
    for ii in range(len(colour_inds)):
        X_train = features[train_set,:][:, colour_inds[ii]]
        Y_train = labels[train_set]
        X_train = StandardScaler().fit_transform(X_train)
#         print(colour_inds[ii],X_train.shape, Y_train.shape)

        lda = LinearDiscriminantAnalysis(store_covariance=True)
        lda.fit(X_train, Y_train).transform(X_train)
        for jj in range(Nsubsets):
            X_test = features[test_subsets[:,jj],:][:, colour_inds[ii]]
            X_test = StandardScaler().fit_transform(X_test)
            Y_test = labels[test_subsets[:,jj]]
            lda_pred = lda.predict(X_test)
            lda_wrong = (abs(lda_pred-Y_test)>0)
#            print("test{}: Nwrong={}, {}, {}".format(jj, lda_wrong.sum(), lda_pred[lda_wrong], Y_test[lda_wrong]))
            accuracy[ii, jj] = 1.-(lda_wrong.sum()/float(X_test.shape[0]))
#     print("Nchoose={}, avg false positive:{}".format(Nchoose, 
#           np.nanmean(accuracy, axis=1)))
    if plot:
        plt.figure(figsize=(6,4))
        plt.hist(np.nanmean(accuracy, axis=1))
        plt.axvline(acc_thresh, ls='--', color='C1', label='Threshold')
        plt.legend()
        plt.xlabel('Avg Accuracy for {} filters'.format(Nchoose))
#     good = (np.nanmean(accuracy, axis=1)>=acc_thresh) 
#     good = good | (np.nanmean(accuracy, axis=1) == np.nanmax(np.nanmean(accuracy, axis=1)))
    if acc_thresh is not None:
        good = (np.nanmean(accuracy, axis=1)>=acc_thresh) 
    else:
        good = np.argsort(np.nanmean(accuracy, axis=1))[::-1][:topX]
#     print("Best filters={}; Avg Accuracy={}; Std={}".format(colour_inds[good], np.nanmean(accuracy, axis=1)[good], 
#                                    np.nanstd(accuracy, axis=1)[good]))
    return colour_inds[good], np.nanmean(accuracy, axis=1)[good], \
            np.nanstd(accuracy, axis=1)[good]
            
def apply_LDA_w_noise(features, labels, train_set, test_subsets, filters):
    """Apply LDA to noisy data (features), given known labels and specified 
    filters/bandpasses and training & test indices."""
    Nsubsets = test_subsets.shape[-1]
    accuracy_w_noise = np.zeros((len(filters), Nsubsets))
    for jj in range(len(filters)):
        X_train = features[train_set,:][:, filters[jj]]
        Y_train = labels[train_set]
        X_train = StandardScaler().fit_transform(X_train)
        lda = LinearDiscriminantAnalysis(store_covariance=True)
        lda.fit(X_train, Y_train).transform(X_train)
        for kk in range(Nsubsets):
            X_test = features[test_subsets[:,kk],:][:, filters[jj]]
            X_test = StandardScaler().fit_transform(X_test)
            Y_test = labels[test_subsets[:,kk]]
            lda_pred = lda.predict(X_test)
            lda_wrong = (abs(lda_pred-Y_test)>0)
            accuracy_w_noise[jj, kk] = 1.-lda_wrong.sum()/float(X_test.shape[0])
    return accuracy_w_noise