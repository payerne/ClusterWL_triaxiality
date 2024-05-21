import sys
import lenspack
from scipy.ndimage.filters import gaussian_filter
from lenspack.utils import bin2d
from scipy.integrate import quad, simps, romb, trapezoid
from lenspack.image.inversion import ks93
from lenspack.geometry.projections.gnom import radec2xy, xy2radec
import numpy as np
from astropy.table import Table
from clmm.utils import compute_lensed_ellipticity
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import simps


def spherical_polar_angle(ra1, dec1, ra2, dec2):
    #from lenspack module
    """Determine the polar angle between two points on the sphere.

    The angle returned is the one subtended between the great circle arc
    joining the two points and the local 'horizontal' axis through the first
    point (ra1, dec1), i.e. dec = dec1.

    Parameters
    ----------
    ra[i], dec[i] : float or array_like
        Coordinates of point [i] on the sphere, where ra[i] is the
        longitudinal angle, and dec[i] is the latitudinal angle in degrees.

    Returns
    -------
    float or numpy array
        Polar angle measured in radians between the dec = dec1 axis and the
        great circle arc joining points [1] and [2].

    Raises
    ------
    Exception
        For inputs of different length.

    Notes
    -----
    The output angle lies within [0, 2 * pi). Multiple second points specified
    by ra2 and dec2 arrays can be given. In that case, the angle of each
    point 2 relative to the single reference point 1 is returned.

    Examples
    --------
    >>> ra1, dec1 = 0, 0  # [deg]
    >>> ra2, dec2 = -1, -1
    >>> angle = spherical_polar_angle(ra1, dec1, ra2, dec2)
    >>> print(np.rad2deg(angle))
    225.00436354465515

    # Deviations from the flat geometric result increase as the points move
    # farther from the equator and their separation gets larger
    >>> ra1, dec1 = 0, 0  # [deg]
    >>> ra2, dec2 = -30, -30
    >>> angle = spherical_polar_angle(ra1, dec1, ra2, dec2)
    >>> print(np.rad2deg(angle))
    229.10660535086907

    """
    # Work in radians
    phi1 = np.deg2rad(np.atleast_1d(ra1))
    theta1 = np.deg2rad(np.atleast_1d(dec1))
    phi2 = np.deg2rad(np.atleast_1d(ra2))
    theta2 = np.deg2rad(np.atleast_1d(dec2))

    # Check input lengths
    if not (len(phi1) == len(theta1)):
        raise Exception("Point 1 array lengths must be the same.")

    if not (len(phi2) == len(theta2)):
        raise Exception("Point 2 array lengths must be the same.")

    if len(phi1) == 1 and len(phi2) > 1:
        phi1 = phi1.repeat(len(phi2))
        theta1 = theta1.repeat(len(theta2))

    # Compute angle
    numerator = np.tan(theta2 - theta1)
    denominator = np.sin((phi2 - phi1) * np.cos(theta1))
    polar_angle = np.arctan2(numerator, denominator)

    # Ensure output range
    negative = polar_angle < 0
    polar_angle[negative] = 2 * np.pi + polar_angle[negative]

    # Clean up
    if len(polar_angle) == 1:
        polar_angle = polar_angle[0]

    return polar_angle

def compute_ellipticity_from_lensing_map(z_cl, z_gal_0, 
                        shear1_map, shear2_map, kappa_map, 
                        lensing_catalog, cosmo = None):
    r"""
    Attributes:
    -----------
    z_cl: float
        cluster redshift
    z_gal_0: float
        default p-300 galaxy redshift
        galaxy redshifts
    shear1_map: fct
        map of shear1
    shear2_map: fct
        map of shear2
    kappa_map: fct
        map of kappa
    lensing_catalog: Table
        Table with ra, dec, z and intrinsic ellipticities
    cosmo: Cosmology object (CLMM)
        cosmology object
    Returns:
    --------
    e1_lensed, e2_lensed: array, array
        lensed ellipticity components 1 & 2
    """
    #store data
    z_gal = lensing_catalog['z']
    ra_gal = lensing_catalog['ra']
    dec_gal = lensing_catalog['dec']
    e1_gal_true = lensing_catalog['e1_true']
    e2_gal_true = lensing_catalog['e2_true']
    
    sigma_crit_z_cl_z_gal_0 = cosmo.eval_sigma_crit(z_cl, z_gal_0)
    sigma_crit_z_cl_z_gal = cosmo.eval_sigma_crit(z_cl, z_gal)
    rescale = sigma_crit_z_cl_z_gal_0/sigma_crit_z_cl_z_gal
    
    #rescaling kappa, shear1, shear2 at given redshifts
    kappa_gal  = kappa_map(ra_gal, dec_gal, grid = False)  * rescale
    shear1_gal = shear1_map(ra_gal, dec_gal, grid = False) * rescale
    shear2_gal = shear2_map(ra_gal, dec_gal, grid = False) * rescale
    
    #compiute lensed ellipticities
    e1_lensed, e2_lensed = compute_lensed_ellipticity(e1_gal_true, e2_gal_true, 
                                                      shear1_gal, shear2_gal, 
                                                      kappa_gal)
    return e1_lensed, e2_lensed

def compute_lensing_map_from_ellipticity(ra_gal, dec_gal, 
                                           e1_gal, e2_gal, 
                                           resolution = .3, 
                                           filter_resolution = None):
    r"""
    Attributes:
    -----------
    ra_gal: array
        galaxy right ascensions
    dec_gal: array
        galaxy declinaisons
    e1_gal: array
        galaxy ellipticity (1st component)
    e2_gal: array
        galaxy ellipticity (2nd component)
    resolution: float
        resolution for ks93 algorithm
    Returns:
    --------
    X, Y: array, array
        right ascension and declinaison of kappas map
    kappaE, kappaB: array, array
        kappa maps from ellipticities
    """
    #ra_mean = np.mean(ra_gal)
    #dec_mean = np.mean(dec_gal)
    # Projection all objects from spherical to Cartesian coordinates
    x, y =  radec2xy(0, 0, ra_gal, dec_gal)
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_x_deg = np.rad2deg(size_x)
    size_y_deg = np.rad2deg(size_y)
    Nx = int(size_x_deg / resolution * 60)
    Ny = int(size_y_deg / resolution * 60)
    x_bin = np.linspace(min_x, max_x, Nx)
    y_bin = np.linspace(min_y, max_y, Nx)
    ra_bin, dec_bin = xy2radec(0, 0, x_bin, y_bin)
    RA_bin = ra_bin
    DEC_bin = dec_bin
    X, Y = np.meshgrid(RA_bin, DEC_bin)
    g1_tmp, g2_tmp = bin2d(x,y, npix=(Nx, Nx), v=(e1_gal, e2_gal), 
                           extent=(min_x, max_x, min_y, max_y))
    g_corr_mc_ngmix_map = np.array([g1_tmp, g2_tmp])
    kappaE, kappaB = ks93(g_corr_mc_ngmix_map[0], -g_corr_mc_ngmix_map[1])
    if filter_resolution != None:
        kappaE = gaussian_filter(kappaE, filter_resolution)
        kappaB = gaussian_filter(kappaB, filter_resolution)
    return X, Y, kappaE, kappaB

def interp_shear_kappa_map(shear1, shear2, kappa, ra, dec):
    r"""
    Attributes:
    -----------
    kappa: array
        tabulated kappa map
    shear1: array
        tabulated shear1 map
    shear2: array
        tabulated shear2 map
    ra: array
        ra axis used for tabulation
    dec: array
        dec axis used for tabulation
    Returns:
    --------
    kappa_map: fct
        interpolated kappa map
    shear1_map: fct
        interpolated shear1 map
    shear2_map: fct
        interpolated shear2 map
    """
    #interpolation of shear1 map
    shear1_map = interp(np.sort(ra), np.sort(dec), shear1)
    #interpolation of shear2 map
    shear2_map = interp(np.sort(ra), np.sort(dec), shear2)
    #interpolation of convergence map
    kappa_map  = interp(np.sort(ra), np.sort(dec), kappa)
    return shear1_map, shear2_map, kappa_map

def multipole_expansion(ra_kappa, dec_kappa, kappa_map, theta_max, m_list = None):
    
    SdRe = {str(m):[] for m in m_list}
    SdIm = {str(m):[] for m in m_list}
    
    theta_ = np.logspace(-7, np.log10(theta_max), 200)
    theta_[0] = 0
    SdIm['theta'] = theta_
    SdRe['theta'] = theta_
    kappa_xy = interp(ra_kappa, dec_kappa, kappa_map)
    phi_axis = np.linspace(0, 2*np.pi, 200)
    
    def kappaf(phi, theta):
        return kappa_xy(theta*np.cos(phi)*180/np.pi, theta*np.sin(phi)*180/np.pi)
    
    for theta in theta_:
        #y = [kappaf(phi[i], theta)[0][0] for i in range(len(phi))]
        kappaf_tab_fixed_theta = [kappaf(phi, theta)[0][0] for phi in phi_axis]
    
        for m in m_list:
            if m == 0: SdRe[str(m)].append(trapezoid(kappaf_tab_fixed_theta, phi_axis)/(2*np.pi))
            else:
                SdRe[str(m)].append(trapezoid(kappaf_tab_fixed_theta * np.cos(-m * phi_axis), phi_axis)/np.pi)
                SdIm[str(m)].append(trapezoid(kappaf_tab_fixed_theta * np.sin(-m * phi_axis), phi_axis)/np.pi)
    Sdmultipole = {'sd_Re':SdRe, 'sd_Im':SdIm}
    return Sdmultipole

def interp(ra_array, dec_array, map2D):
    r"""
    Attributes:
    -----------
    ra_array, dec_array : array, array
        ra, dec axis of the map
    map: array
        2D map
    Returns:
    --------
    fct : function
        interpolated 2D map
    """
    return RectBivariateSpline(ra_array, dec_array, map2D)

# def moments(ra_kappa, dec_kappa, kappa_map, theta_max):

#     theta_ = np.logspace(-4, np.log10(theta_max), 80)
#     theta_[0] = 0
#     kappa_xy = interp(ra_kappa, dec_kappa, kappa_map)
#     phi = np.linspace(0, 2*np.pi, 100)
#     def kappaf(phi, theta):
#         return kappa_xy(theta*np.cos(phi)*180/np.pi, theta*np.sin(phi)*180/np.pi)
#     sigma0 = []
#     sigma2 = []
#     for theta in theta_:
#         y = [kappaf(phi[i], theta)[0][0] for i in range(len(phi))]
#         sigma0.append(simps(y, phi)/(2*np.pi))
#         sigma2.append(simps(y * np.cos(2*phi), phi)/(np.pi))
#     sigma2 = np.array(sigma2)
#     sigma0 = np.array(sigma0)
#     x_2, t_2, t = [], [], []
#     for theta in theta_[1:]:
#         mask = theta_ < theta
#         mask_up = np.invert(mask)
#         alpha_1 = simps(sigma2[mask_up]*theta_[mask_up]**(-1), theta_[mask_up])
#         alpha3 = simps(sigma2[mask]*theta_[mask]**3, theta_[mask])
#         alpha =  simps(sigma0[mask]*theta_[mask], theta_[mask])

#         x_2.append(alpha3*3/(theta**4) - alpha_1)

#         t_2.append(- np.interp(theta, theta_, sigma2) + alpha3*3/(theta**4) + alpha_1)

#         t.append(alpha*2/theta**2 - np.interp(theta, theta_, sigma0))

#     return theta_, sigma0, sigma2, np.array(t), np.array(t_2), np.array(x_2)




def surface_density_mutlipoles(ra_kappa, dec_kappa, kappa_map, theta_max, trigo = None, m_list = None):
    
    theta_ = np.logspace(-7, np.log10(theta_max), 100)
    theta_[0] = 0
    kappa_xy = interp(ra_kappa, dec_kappa, kappa_map)
    phi = np.linspace(0, 2*np.pi, 500)
    def kappaf(phi, theta):
        return kappa_xy(theta*np.cos(phi)*180/np.pi,theta*np.sin(phi)*180/np.pi)
    multipole_sigma = {'Sigma'+str(m_):[] for m_ in m_list}
    for i, theta in enumerate(theta_):
        if theta > 0:
            y = [kappaf(phi[i], theta)[0][0] for i in range(len(phi))]
        elif theta == 0:
            y = [kappaf(0, theta)[0][0] for i in range(len(phi))]
        for m in m_list:
            if m==0:
                multipole_sigma['Sigma'+str(m)].append(simps(y, phi)/(2*np.pi))
            else: 
                if trigo == 'cos':
                    multipole_sigma['Sigma'+str(m)].append(simps(y * np.cos(m*phi), phi)/(np.pi))
                elif trigo == 'sin':
                    multipole_sigma['Sigma'+str(m)].append(simps(y * np.sin(m*phi), phi)/(np.pi))
    for m in m_list:
        multipole_sigma['Sigma'+str(m)] = np.array(multipole_sigma['Sigma'+str(m)])
    multipole_sigma = Table(multipole_sigma)
    multipole_sigma['theta'] = theta_
    
    return multipole_sigma

# def tangential_shear_multipoles(multipole_sigma, m_list = None):

#     multipole_tshear = {'+shear'+str(m_):[] for m_ in m_list}
#     x_2, t_2, t = [], [], []
#     theta_m = multipole_sigma['theta']
#     theta_eval = np.logspace(-5, np.log10(max(theta_m)*.95), 30)
#     #theta_shear[0] = 0
#     for m in m_list:
#         sigmam = multipole_sigma['Sigma'+str(m)]
#         def sigmamf(x): return np.interp(x, theta_m, sigmam)
#         print(sigmam.shape, theta_m.shape)
#         for i, theta in enumerate(theta_eval):
#             mask_down = theta_m <= theta
#             theta_down = theta_m[mask_down]
#             mask_up = np.invert(mask_down)
#             theta_up = theta_m[mask_up]
#             if m > 0:
#                 #alpha^up_(1-m)
#                 sigmamup = sigmamf(theta_up)
#                 up = simps(sigmamup*theta_up**(1-m), theta_up)
#                 #alpha^down_(m+1)
#                 sigmamdown = sigmamf(theta_down)
#                 down = simps(sigmamdown*theta_down**(m+1), theta_down)
#                 #sigmam
#                 sigmam_ = sigmamf(theta)
#                 sheartm = -sigmam_ + (m+1)*down*theta**(-(m+2)) + (m-1)*up*theta**(m-2)
#             else:
#                 sigmamdown = sigmamf(theta_down)
#                 a2 = simps(sigmamdown*theta_down, theta_down)
#                 a1 = sigmamf(theta)
#                 sheartm = -a1 + a2*2/(theta**2)
#             multipole_tshear['+shear'+str(m)].append(sheartm)
#     for m in m_list:
#         multipole_tshear['+shear'+str(m)] = np.array(multipole_tshear['+shear'+str(m)])
#     multipole_tshear = Table(multipole_tshear)
#     multipole_tshear['theta'] = theta_eval
#     return multipole_tshear

# def oldtangential_shear_multipoles(multipole_sigma, m_list = None):

#     multipole_tshear = {'+shear'+str(m_):[] for m_ in m_list}
#     x_2, t_2, t = [], [], []
#     theta_ = multipole_sigma['theta']
#     theta_shear = np.logspace(-7, max(theta_), 150)
#     for m in m_list:
#         sigmam = multipole_sigma['Sigma'+str(m)]
#         for i, theta in enumerate(theta_[1:]):
#             mask_down = theta_ < theta
#             mask_up = np.invert(mask_down)
#             if m > 0:
#                 def sigmam(x):
#                     return np.interp(x, theta_, sigmam)
#                 #alpha^up_(1-m)
#                 up = simps(sigmam[mask_up]*theta_[mask_up]**(1-m), theta_[mask_up])
#                 #alpha^down_(m+1)
#                 down = simps(sigmam[mask_down]*theta_[mask_down]**(m+1), theta_[mask_down])
#                 #sigmam
#                 sigmam_ = np.interp(theta, theta_, sigmam)
#                 sheartm = -sigmam_ + (m+1)*down*theta**(-(m+2)) + (m-1)*up*theta**(m-2)
#             else:
#                 a2 = simps(sigmam[mask_down]*theta_[mask_down], theta_[mask_down])
#                 a1 = np.interp(theta, theta_, sigmam)
#                 sheartm = -a1 + a2*2/(theta**2)
#             multipole_tshear['+shear'+str(m)].append(sheartm)
#     for m in m_list:
#         multipole_tshear['+shear'+str(m)] = np.array(multipole_tshear['+shear'+str(m)])
#     multipole_tshear = Table(multipole_tshear)
#     multipole_tshear['theta'] = theta_[1:]
#     return multipole_tshear

