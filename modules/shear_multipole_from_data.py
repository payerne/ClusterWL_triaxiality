import numpy as np
from astropy.table import Table, hstack

def make_multipoles_profiles(galaxycatalog, radial_bins, m_list=None):
    r"""
    Attributes:
    -----------
    galaxycatalog: Table
        background galaxy catalog (with tangential and cross components, z_gal, radius)
    radial_bins: list
        list of radial bins
    cosmo: Object
        CLMM cosmology
    m_list: list
        list of multipoles
    Returns:
    --------
    profiles: Table
        Table of estimated shear multipoles
    """
    name = ['sheart','shearx']
    trigo = ['Re','Im']
    prf = {name_ + '_' + trigo_ + '_'+str(m_):[] for name_ in name for trigo_ in trigo for m_ in m_list}
    prf_std = {'std_' + name_ + '_' + trigo_ + '_'+str(m_):[] for name_ in name for trigo_ in trigo for m_ in m_list}
    
    def f(trigo_, m, phi):
        if trigo_ =='Re': return np.cos(-m * phi)
        if trigo_ =='Im': return np.sin(-m * phi)
    
    def comp(name_, et, ex):
        if name_=='sheart': return et
        if name_=='shearx': return ex
    
    for i, b in enumerate(radial_bins):
        
        mask = (galaxycatalog['R'] > b[0])*(galaxycatalog['R'] < b[1])
        data_binned = galaxycatalog[mask]
        phi = data_binned['phi']
        wgeo_ls = data_binned['wgeo_ls']
        
        #ellipticity components
        et = data_binned['et']
        ex = data_binned['ex']
        
        for i, name_ in enumerate(name):
            for j, trigo_ in enumerate(trigo):
                for k, m_ in enumerate(m_list):
                    if (m_==0)*(trigo_=='Im'):
                        #since sin(m*phi)=0
                        prf[name_ + '_' + trigo_ + '_'+str(m_)].append(0)
                    else:
                        w_esd = wgeo_ls*f(trigo_, m_, phi)
                        esd_unnormed = np.sum(comp(name_, et, ex) * w_esd)
                        esd_ = esd_unnormed/np.sum(wgeo_ls * f(trigo_, m_, phi)**2)
                        prf[name_ + '_' + trigo_ + '_'+str(m_)].append(esd_)
                        #
                        esd_std = np.average((comp(name_, et, ex) - esd_)**2, weights = None)**.5 
                        esd_mean_std = esd_std/np.sqrt(len(et))
                        prf_std['std_'+name_ + '_' + trigo_ + '_'+str(m_)].append(esd_mean_std)
                        
    t = Table(prf)
    t['radius'] = np.mean(radial_bins, axis=1)
    
    return t, prf_std

def compute_multipole_covariance(galaxycatalog, radial_bins, n_boot=100, m_list=None):
    r"""
    Attributes:
    -----------
    galaxycatalog: Table
        background galaxy catalog (with tangential and cross components, z_gal, radius)
    radial_bins: list
        list of radial bins
    cosmo: Object
        CLMM cosmology
    n_boot: int
        number of bootstrap resampling
    m_list: list
        list of multipoles
    Returns:
    --------
    full_covariance: array
        joint covariance of multipoles
    individual covariance: dict
        dictionary of shear multipole individual covariances
    """
    name_shear = ['sheart', 'shearx']
    trigo = ['Re', 'Im']
    names = [name_ + '_' + trigo_ + '_'+str(m_) for name_ in name_shear for trigo_ in trigo for m_ in m_list]
    
    n_profiles = len(m_list) * 2 * 2
    n_radius = len(radial_bins)
    arr = np.zeros([n_boot, n_profiles*n_radius])
    n_gal = len(galaxycatalog)
    for i in range(n_boot):
        #select galaxies in angular bins
        index_gal = np.arange(n_gal)
        index_boot = np.random.choice(index_gal, size=n_gal, replace=True)
        data_cut = galaxycatalog[index_boot]
        prf_boot = make_multipoles_profiles(data_cut, radial_bins, m_list=m_list)
        add = []
        for n in names:
            add.extend(list(prf_boot[n]))
        arr[i,:] = np.array(add)
    Xt_boot = np.stack(arr.astype(float), axis = 1)
    full_covariance = np.cov(Xt_boot, bias = False)
    
    cov_individual = {}
    for i, n in enumerate(names):
            cov_individual[n]=full_covariance[i*n_radius:(i+1)*n_radius, i*n_radius:(i+1)*n_radius]
    
    return full_covariance, cov_individual

