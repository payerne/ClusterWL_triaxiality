import numpy as np
import scipy
from scipy.integrate import quad, simps, romb, trapezoid
from scipy.interpolate import interp1d
from astropy.table import Table

def excess_surface_density_multipoles(Rproj, Sdmultipoles, m_list = None):
    r"""
    Attributes:
    -----------
    Rproj: array
        radius from the cluster center
    Sdmultipoles: dict
        dictionnary of surface density multipoles
    m_list: list
        list of multipoles
    Returns:
    --------
    ESSdmultipoles: dict
        dictionnary of excess surface density multipoles
    """

    Sheart_Re = {'sheart_Re_'+str(m):[] for m in m_list}
    Sheart_Im = {'sheart_Im_'+str(m):[] for m in m_list}
    Shearx_Re = {'shearx_Re_'+str(m):[] for m in m_list}
    Shearx_Im = {'shearx_Im_'+str(m):[] for m in m_list}
    
    SdRe = Sdmultipoles['sd_Re']
    SdIm = Sdmultipoles['sd_Im']
    r = Sdmultipoles['sd_Re']['r']

    for m in m_list:

        sdRem = np.array(SdRe[str(m)])
        sdImm = np.array(SdIm[str(m)])
        
        def integrand(x, m, f_tab):
                return x ** (m+1) * np.interp(x, r, f_tab, left=f_tab[0], right=f_tab[-1])
            
        def Gammam_fct(R, m):
            
            x_in_R = np.linspace(1e-7, R, 50)
            x_out_R = np.linspace(R, max(r), 50)
            
            #integrand_in_Rem = integrand(x_in_R, m, sdRem)
            #integrand_out_Rem = integrand(x_out_R, m, sdRem)
            
            #integrand_in_Imm = integrand(x_in_R, m, sdImm)
            #integrand_out_Imm = integrand(x_out_R, m, sdImm)
            
            #xout = r[r>R]
            #xin  = r[r<R]
            
            if m >= 0:
                #Re = -np.interp(R, r, sdRem) + 2*(m+1)*R**(-(m+2))*simps(integrand(xin, m, sdRem), xin)
                #Im = -np.interp(R, r, sdImm) + 2*(m+1)*R**(-(m+2))*simps(integrand(xin, m, sdImm), xin)
                sdRemplus = sdRem
                sdImmplus = sdImm
                integrand_in_Rem = integrand(x_in_R, m, sdRemplus)
                integrand_in_Imm = integrand(x_in_R, m, sdImmplus)
                Re = -np.interp(R, r, sdRemplus) + 2*(m+1)*R**(-(m+2))*trapezoid(integrand_in_Rem, x_in_R)
                Im = -np.interp(R, r, sdImmplus) + 2*(m+1)*R**(-(m+2))*trapezoid(integrand_in_Imm, x_in_R)
               
            if m < 0:
                #Re = -np.interp(R, r, sdRem) - 2*(m+1)*R**(-(m+2))*simps(integrand(xout, m, sdRem), xout)
                #Im = -np.interp(R, r, sdRem) - 2*(m+1)*R**(-(m+2))*simps(integrand(xout, m, sdImm), xout)
                sdRemminus = sdRem
                sdImmminus = -sdImm
                integrand_out_Rem = integrand(x_out_R, m, sdRemminus)
                integrand_out_Imm = integrand(x_out_R, m, sdImmminus)
                Re = -np.interp(R, r, sdRemminus) - 2*(m+1)*R**(-(m+2))*trapezoid(integrand_out_Rem, x_out_R)
                Im = -np.interp(R, r, sdImmminus) - 2*(m+1)*R**(-(m+2))*trapezoid(integrand_out_Imm, x_out_R)
            
            return Re +1j*Im

        if m > 0:
            for i, Ri in enumerate(Rproj):

                Gammam  = Gammam_fct(Ri,  m)
                Gamma_m = Gammam_fct(Ri, -m)
                Sheart_Re['sheart_Re_'+str(m)].append((Gammam.real+Gamma_m.real)/2)
                Sheart_Im['sheart_Im_'+str(m)].append((Gammam.imag-Gamma_m.imag)/2)
                Shearx_Re['shearx_Re_'+str(m)].append((Gammam.imag+Gamma_m.imag)/2)
                Shearx_Im['shearx_Im_'+str(m)].append(-(Gammam.real-Gamma_m.real)/2)
    
        elif m==0:
            for i, Ri in enumerate(Rproj):
                #xdown = r[r<Ri]
                a1 = np.interp(Ri, r, sdRem)
                #a2 = simps(integrand(xdown, m, sdRem), xdown)*2/(Ri**2)
                x_in_R = np.linspace(1e-15, Ri, 100)
                a2 = trapezoid(integrand(x_in_R, 0, sdRem), x_in_R)*2/(Ri**2)
                Sheart_Re['sheart_Re_'+str(m)].append(-a1 + a2)
                Sheart_Im['sheart_Im_'+str(m)].append(0)
                Shearx_Re['shearx_Re_'+str(m)].append(0)
                Shearx_Im['shearx_Im_'+str(m)].append(0)
    
    full = {}
    for m in m_list:
        full['sheart_Re_'+str(m)] = Sheart_Re['sheart_Re_'+str(m)]
        full['sheart_Im_'+str(m)] = Sheart_Im['sheart_Im_'+str(m)]
        full['shearx_Re_'+str(m)] = Shearx_Re['shearx_Re_'+str(m)]
        full['shearx_Im_'+str(m)] = Shearx_Im['shearx_Im_'+str(m)]
    full['R'] = Rproj
        
    return full

def surface_density_spherical(z, logm, c, moo, convergence = False):
    
    moo.set_mass(10**logm)
    moo.set_concentration(c)
    
    r0 = .1e-1
    if convergence==True:
        sdr0 = moo.eval_convergence(r0, z, 3)
    else:     
        sdr0 = moo.eval_surface_density(r0, z)
    
    def Sd_spherical(r):
        if convergence==True:
            sd = moo.eval_convergence(r, z, 3)
        else:
            sd = moo.eval_surface_density(r, z)
       
        mask = r < r0
        #sd[mask] = sdr0
        return sd
    
    return Sd_spherical

def surface_density_multipoles(surface_density_spherical, ax, qxy, phi0, m_list = None):
    r"""
    Attributes:
    -----------
    z: float
        cluster redshift
    Returns:
    --------
    SDdmultipoles: dict
        dictionnary of surface density multipoles
    """

    a = ax
    b = qxy*a
    n_R = 2500
    n_phi = 250

    sd = np.zeros([n_R, n_phi])
    phi = np.linspace(0, 2*np.pi, n_phi)
    r = np.logspace(np.log10(0.0001), np.log10(30), n_R)
    r[0] = 1e-15
    Phi_, R_ = np.meshgrid(phi, r)
    R_flat, Phi_flat = R_.flatten(), Phi_.flatten()
    a_term = np.cos(Phi_flat-phi0)
    b_term = np.sin(Phi_flat-phi0)
    a_eq_flat = np.sqrt((a_term/a)**2 + (b_term/b)**2)

    Sd_flat = surface_density_spherical(R_flat * a_eq_flat)/(a*b)
    Sd = Sd_flat.reshape([n_R, n_phi])
    SdRe = {str(m):[] for m in m_list}
    SdRe['r'] = r
    SdIm = {str(m):[] for m in m_list}
    SdIm['r'] = r
    for m in m_list:
        if m == 0:
            SdRe[str(m)] = simps(Sd, phi, axis = 1)/(2*np.pi)
        else:
            SdRe[str(m)] = simps(Sd * np.cos(-m * Phi_), phi, axis = 1)/np.pi
            SdIm[str(m)] = simps(Sd * np.sin(-m * Phi_), phi, axis = 1)/np.pi
    
    Sdmultipole = {'sd_Re':SdRe, 'sd_Im':SdIm}
    #the complex surface density multipole is given by Sdm = SdRem + i*SdImm
        
    return Sdmultipole

def surface_density_multipoles_adhikari(surface_density_spherical, ax, qxy, phi0, m_list = None):
    
    a = ax
    b = qxy*a
    n_R = 2000
    n_phi = 300
    sd = np.zeros([n_R, n_phi])
    phi = np.linspace(0, 2*np.pi, n_phi)
    r = np.logspace(np.log10(0.001), np.log10(20), n_R)
    r[0] = 1e-9
    Phi_, R_ = np.meshgrid(phi, r)
    R_flat, Phi_flat = R_.flatten(), Phi_.flatten()
    a_term = np.cos(Phi_flat)
    b_term = np.sin(Phi_flat)
    a_eq_flat = np.sqrt((a_term/a)**2 + (b_term/b)**2)
    
    r0 = .1e-1
    sdr0 = moo.eval_surface_density(r0, z)
    def Sd_spherical(r):
        sd = moo.eval_surface_density(r, z)
        mask = r < r0
        sd[mask] = sdr0
        return sd

    Sd_flat = Sd_spherical(R_flat * a_eq_flat)/(a*b)
    Sd = Sd_flat.reshape([n_R, n_phi])
    Sdcosm = {'Sd'+str(m):[] for m in m_list}
    Sdcosm['r'] = r
    Sdsinm = {'Sd'+str(m):[] for m in m_list}
    Sdsinm['r'] = r
    Sdcosm['Sd0'] = simps(Sd, phi, axis = 1)/(2*np.pi)
    
    #compute approximation from
    ellipticity = (1 -qxy**2)/(2*(1+qxy**2))
    lnSd0 = np.log(Sdcosm['Sd0'])
    lnr = np.log(Sdcosm['r'])
    eta0 = -np.gradient(lnSd0, lnr)
    delta = (eta0/2)*(1 + eta0/2)
    Sd0 = Sdcosm['Sd0']
    A = Sd0/(1 + ellipticity**2 * delta)
    Sdcosm['Sd2'] = A * ellipticity * eta0 * np.cos(2*phi0)
    Sdcosm['Sd4'] = A * ellipticity **2 * delta * np.cos(4*phi0)
    Sdsinm['Sd0'] = 0 * Sd0
    Sdsinm['Sd2'] = - A * ellipticity * eta0 * np.sin(2*phi0)
    Sdsinm['Sd4'] = - A * ellipticity **2 * delta * np.sin(4*phi0)
    
    return {'Sdcosm':Sdcosm, 'Sdsinm':Sdsinm}

def sigmac_alpha(z_cl, alpha, z_distrib, cosmo):
    r"""
    Attributes:
    -----------
    z_cl: float
        cluster redshift
    alpha: int
        power exponent
    cosmo: Cosmology
        Cosmology object of CLMM
    Returns:
    --------
    sigmac_alpha: float
        average (sigmac)**(alpha) over Chang et al. redshift pdf
    """
    z_axis = np.linspace(z_cl, 100, 5000)
    sigmac = cosmo.eval_sigma_crit(z_cl, z_axis)
    av_sigmac_alpha = simps( sigmac**alpha * z_distrib(z_axis), z_axis) 
    return av_sigmac_alpha

# def elliptical_esd(R, z, logm, c, ax, qxy, phi0, cosmo, moo, 
#                    esd=False, esd_2=False, esd_x_2=False, 
#                    method='tab', m_list = None):
#     r"""
#     Attributes:
#     -----------
#     R: array
#         radius (Mpc)
#     z: float
#         redshift
#     logm: float
#         mass (log10)
#     c: float
#         concentration
#     ax: float
#         axis deformation (x)
#     qxy: float
#         axis ratio
#     phi0: float
#         orientation (0-np.pi)
#     cosmo: Cosmology
#         CLMM cosmology object
#     moo: Modeling
#         CLMM modeling object
#     esd: Bool
#         compute esd
#     esd_2: Bool
#         compute esd_2
#     esd_x_2: Bool
#         compute_esd_x_2
#     method: str
#         method to use
#     """

#     moo.set_cosmo(cosmo)
#     moo.set_mass (10**logm)
#     moo.set_concentration (c)

#     a = ax
#     b = qxy*a

#     if method=='interp':
#         #modeling of spherical profile
#         def shear_order0(R, convergence_order0):
#         #return shear (order 0)
#             def integrand(x):
#                 return x*convergence_order0(x)
#             res = []
#             for R_ in R:
#                 a1 = - convergence_order0(R_)
#                 a2 = quad(integrand, 1e-7, R_)[0]*2/(R_**2)
#                 res.append(a1+a2)
#             return np.array(res)

#         def shear_order2(R, convergence_order2):
#             #return shear (quadrupole)
#             res = []
#             def __integrand__1(R): return R ** 3 * convergence_order2(R)
#             def __integrand__2(R): return R ** (-1) * convergence_order2(R)
#             for R_ in R:
#                 a1 = -convergence_order2(R_)
#                 a2 = (3/(R_**4)) * quad(__integrand__1, 1e-7, R_)[0]
#                 a3 = quad(__integrand__2, R_, 40)[0]
#                 res.append(a1 + a2 + a3)
#             return np.array(res)

#         def shear_x_order2(R, convergence_order2):
#              #return cross shear (quadrupole)
#             res = []
#             def __integrand__1(R): return R ** 3 * convergence_order2(R)
#             def __integrand__2(R): return R**(-1) * convergence_order2(R)
#             for R_ in R:
#                 a1 = (3/(R_**4)) * quad(__integrand__1, 1e-7, R_)[0]
#                 a2 = -quad(__integrand__2, R_, 40)[0]
#                 res.append(a1 + a2)
#             return np.array(res)

#         def polar_average(fct, R):
#             r"""
#             Attributes:
#             -----------
#             spherical: fct
#                 function returning the radial profile
#             R: array
#                 radius
#             q: float
#                 axis ratio < 1
#             Returns:
#             --------
#             res: array
#                 elliptical averaged profile at R
#             """
#             res = []
#             phi = np.linspace(0, 2*np.pi, 3000)
#             for R_ in R:
#                 __integrand__phi = fct(phi, R_)
#                 integral = simps(__integrand__phi, phi)
#                 res.append(integral)
#             res =np.array(res)
#             return res

#         x_tab0 = np.logspace(-10, 2, 500)
#         x_tab1 = np.logspace(-9, 1.7, 500)
#         y_tab0 =moo.eval_surface_density(x_tab0, z, )

#         #interp sph convergence 
#         def convergence(x):
#             return np.interp(x, x_tab0, y_tab0)

#         #ell convergence
#         def convergence_elliptical_0_phiR(phi, R):
#             R_eff = R*np.sqrt((np.cos(phi-phi0)/a)**2 + (np.sin(phi-phi0)/b)**2)
#             return convergence(R_eff)/(a*b)

#         #polar average ell convergence
#         if esd==True:
#             convergence_elliptical_0_tab = polar_average(convergence_elliptical_0_phiR, x_tab1)/(2*np.pi)

#             #interp polar average ell convergence
#             def convergence_elliptical_0(x): return np.interp(x, x_tab1, convergence_elliptical_0_tab)
#             convergence_elliptical_0 = scipy.interpolate.interp1d(x_tab1, convergence_elliptical_0_tab,
#                                                                 kind='cubic', axis=- 1, copy=False, bounds_error=True, fill_value=0)
#         #ell convergence (2)
#         if esd_2==True or (esd_x_2==True):
#             def convergence_elliptical_2_phiR(phi, R):
#                 R_eff = R*np.sqrt((np.cos(phi-phi0)/a)**2 + (np.sin(phi-phi0)/b)**2)
#                 return (convergence(R_eff)/(a*b))*np.cos(2*phi)

#             #polar average ell convergence (2)
#             convergence_elliptical_2_tab = polar_average(convergence_elliptical_2_phiR, x_tab1)/np.pi
#             def convergence_elliptical_2(x): return np.interp(x, x_tab1, convergence_elliptical_2_tab)

#             #interp polar average ell convergence (2)
#             convergence_elliptical_2 = scipy.interpolate.interp1d(x_tab1, convergence_elliptical_2_tab,
#                                                                 kind='cubic', axis=- 1, copy=False, bounds_error=True, fill_value=0)
#         res = []
#         if esd == True: res.append(shear_order0(R, convergence_elliptical_0))
#         if esd_2 == True: res.append(shear_order2(R, convergence_elliptical_2))
#         if esd_x_2 == True: res.append(shear_x_order2(R, convergence_elliptical_2))
#         return res

#     if method=='tab':

#         n_R = 4000
#         n_phi = 400

#         sd = np.zeros([n_R, n_phi])
#         phi = np.linspace(0, 2*np.pi, n_phi)
#         r = np.logspace(np.log10(0.0001), np.log10(20), n_R)
#         r[0] = 1e-9
#         Phi_, R_ = np.meshgrid(phi, r)
#         R_flat, Phi_flat = R_.flatten(), Phi_.flatten()
#         #a_eq_flat = np.sqrt((np.cos(Phi_flat-phi0)/a)**2 + (np.sin(Phi_flat-phi0)/b)**2)
#         a_term = np.cos(Phi_flat-phi0)#np.cos(Phi_flat)*np.cos(phi0) + np.sin(Phi_flat)*np.sin(phi0)
#         b_term = np.sin(Phi_flat-phi0)#np.cos(Phi_flat)*np.sin(phi0) - np.sin(Phi_flat)*np.cos(phi0)
#         a_eq_flat = np.sqrt((a_term/a)**2 + (b_term/b)**2)

#         r0 = .1e-2
#         sdr0 = moo.eval_surface_density(r0, z)
#         def Sd_spherical(r):
#             sd = moo.eval_surface_density(r, z)
#             mask = r < r0
#             sd[mask] = sdr0
#             return sd

#         Sd_flat = Sd_spherical(R_flat * a_eq_flat)/(a*b)
#         Sd = Sd_flat.reshape([n_R, n_phi])
#         Sdcosm = {'Sd'+str(m):[] for m in m_list}
#         Sdcosm['r'] = r
#         Sdsinm = {'Sd'+str(m):[] for m in m_list}
#         Sdsinm['r'] = r
#         def trigo(x): return np.cos(x)
#         for m in m_list:
#             if m == 0:
#                 Sdcosm['Sd'+str(m)] = simps(Sd, phi, axis = 1)/(2*np.pi)
#             else:
#                 Sdcosm['Sd'+str(m)] = simps(Sd * np.cos(-m * Phi_), phi, axis = 1)/np.pi
#                 Sdsinm['Sd'+str(m)] = simps(Sd * np.sin(-m * Phi_), phi, axis = 1)/np.pi

#         Sheart = {'+shear_'+str(m):[] for m in m_list}
#         Shearx = {'xshear_'+str(m):[] for m in m_list}

#         for m in m_list:

#             sdcosm = Sdcosm['Sd'+str(m)]
#             sdsinm = Sdsinm['Sd'+str(m)]
#             def integrand(x, ms, sd):
#                     return x * x**ms * np.interp(x, r, sd)

#             if m > 0:
#                 for i, Ri in enumerate(R):
#                     a1 = np.interp(Ri, r, sdcosm)
#                     xout = r[r>Ri]
#                     xin = r[r<Ri]
#                     a2 = simps(integrand(xin, m, sdcosm), xin)
#                     a3 = simps(integrand(xout,  -m, sdcosm), xout)

#                     Sheart['+shear_'+str(m)].append(-a1 + (m+1)*Ri**(-(2+m))*a2 - (-m+1)*Ri**(-(2-m))*a3)

#                     a2 = simps(integrand(xin, m, sdsinm), xin)
#                     a3 = simps(integrand(xout,-m, sdsinm), xout)
#                     Shearx['xshear_'+str(m)].append((m+1)*Ri**(-m-2)*a2 - (m-1)*Ri**(m-2)*a3)
#             elif m==0:
#                 for i, Ri in enumerate(R):
#                     xin = r[r<Ri]
#                     a1 = np.interp(Ri, r, sdcosm)
#                     a2 = simps(integrand(xin, m, sdcosm), xin)*2/(Ri**2)
#                     Sheart['+shear_'+str(m)].append(-a1 + a2)
#                     Shearx['xshear_'+str(m)].append(0)

#         return Sd, Sdcosm, Sheart, Shearx


# def elliptical_esd_2(R, z, logm, c, ax, qxy, phi0, cosmo, moo, 
#                    esd=False, esd_2=False, esd_x_2=False, 
#                    method='tab', m_list = None):

#     moo.set_cosmo(cosmo)
#     moo.set_mass (10**logm)
#     moo.set_concentration (c)

#     a = ax
#     b = qxy*a
#     n_R = 2000
#     n_phi = 300

#     sd = np.zeros([n_R, n_phi])
#     phi = np.linspace(0, 2*np.pi, n_phi)
#     r = np.logspace(np.log10(0.001), np.log10(20), n_R)
#     r[0] = 1e-9
#     Phi_, R_ = np.meshgrid(phi, r)
#     R_flat, Phi_flat = R_.flatten(), Phi_.flatten()
#     a_term = np.cos(Phi_flat-phi0)
#     b_term = np.sin(Phi_flat-phi0)
#     a_eq_flat = np.sqrt((a_term/a)**2 + (b_term/b)**2)

#     r0 = .1e-2
#     sdr0 = moo.eval_surface_density(r0, z)
#     def Sd_spherical(r):
#         sd = moo.eval_surface_density(r, z)
#         mask = r < r0
#         sd[mask] = sdr0
#         return sd

#     Sd_flat = Sd_spherical(R_flat * a_eq_flat)/(a*b)
#     Sd = Sd_flat.reshape([n_R, n_phi])
#     Sdcosm = {'Sd'+str(m):[] for m in m_list}
#     Sdcosm['r'] = r
#     Sdsinm = {'Sd'+str(m):[] for m in m_list}
#     Sdsinm['r'] = r
#     for m in m_list:
#         if m == 0:
#             Sdcosm['Sd'+str(m)] = simps(Sd, phi, axis = 1)/(2*np.pi)
#         else:
#             Sdcosm['Sd'+str(m)] = simps(Sd * np.cos(-m * Phi_), phi, axis = 1)/np.pi
#             Sdsinm['Sd'+str(m)] = simps(Sd * np.sin(-m * Phi_), phi, axis = 1)/np.pi

#     Sheart_cos = {'+shear_cos_'+str(m):[] for m in m_list}
#     Sheart_sin = {'+shear_sin_'+str(m):[] for m in m_list}
#     Shearx_cos = {'xshear_cos_'+str(m):[] for m in m_list}
#     Shearx_sin = {'xshear_sin_'+str(m):[] for m in m_list}

#     for m in m_list:

#         sdcosm = np.array(Sdcosm['Sd'+str(m)])
#         sdsinm = np.array(Sdsinm['Sd'+str(m)])
#         def integrand(x, m, sd):
#                 return x * x ** m * np.interp(x, r, sd)

#         if m > 0:
#             for i, Ri in enumerate(R):

#                 xout = r[r>Ri]
#                 xin  = r[r<Ri]
#                 Gammam_Re  = -np.interp(Ri, r, sdcosm)  + 2*(m+1)*Ri**(-(2+m))*simps(integrand(xin, m, sdcosm), xin)
#                 Gammam_Im  = -np.interp(Ri, r, sdsinm)  + 2*(m+1)*Ri**(-(2+m))*simps(integrand(xin, m, sdsinm), xin)
#                 Gamma_m_Re = -np.interp(Ri, r, sdcosm)  - 2*(-m+1)*Ri**(-(2-m))*simps(integrand(xout,-m,  sdcosm), xout)
#                 Gamma_m_Im = -np.interp(Ri, r, -sdsinm) - 2*(-m+1)*Ri**(-(2-m))*simps(integrand(xout,-m, -sdsinm), xout)
#                 Sheart_cos['+shear_cos_'+str(m)].append((Gammam_Re+Gamma_m_Re)/2)
#                 Sheart_sin['+shear_sin_'+str(m)].append(-(Gammam_Im-Gamma_m_Im)/2)
#                 Shearx_cos['xshear_cos_'+str(m)].append((Gammam_Im+Gamma_m_Im)/2)
#                 Shearx_sin['xshear_sin_'+str(m)].append((Gammam_Re-Gamma_m_Re)/2)

#         elif m==0:
#             for i, Ri in enumerate(R):
#                 xdown = r[r<Ri]
#                 a1 = np.interp(Ri, r, sdcosm)
#                 a2 = simps(integrand(xdown, m, sdcosm), xdown)*2/(Ri**2)
#                 Sheart_cos['+shear_cos_'+str(m)].append(-a1 + a2)
#                 Sheart_sin['+shear_sin_'+str(m)].append(0)
#                 Shearx_cos['xshear_cos_'+str(m)].append(0)
#                 Shearx_sin['xshear_sin_'+str(m)].append(0)

#     full = {}
#     for m in m_list:
#         full['+shear_cos_'+str(m)] = Sheart_cos['+shear_cos_'+str(m)]
#         full['+shear_sin_'+str(m)] = Sheart_sin['+shear_sin_'+str(m)]
#         full['xshear_cos_'+str(m)] = Shearx_cos['xshear_cos_'+str(m)]
#         full['xshear_sin_'+str(m)] = Shearx_sin['xshear_sin_'+str(m)]

#     return Table(full), Sdcosm, Sdsinm

    
    
    
    
    # def elliptical_esd_from_sd(Rproj, Sdmultipole, m_list = None):

#     Sheart_cos = {'+shear_cos_'+str(m):[] for m in m_list}
#     Sheart_sin = {'+shear_sin_'+str(m):[] for m in m_list}
#     Shearx_cos = {'xshear_cos_'+str(m):[] for m in m_list}
#     Shearx_sin = {'xshear_sin_'+str(m):[] for m in m_list}

#     Sdcosm = Sdmultipole['Sdcosm']
#     Sdsinm = Sdmultipole['Sdsinm']
#     r = Sdmultipole['Sdcosm']['r']

#     for m in m_list:

#         sdcosm = np.array(Sdcosm['Sd'+str(m)])
#         sdsinm = np.array(Sdsinm['Sd'+str(m)])
#         def integrand(x, m, sd):
#                 return x * x ** m * np.interp(x, r, sd)

#         if m > 0:
#             for i, Ri in enumerate(Rproj):

#                 xout = r[r>Ri]
#                 xin  = r[r<Ri]
#                 Gammam_Re  = -np.interp(Ri, r, sdcosm)  + 2*(m+1)*Ri**(-(2+m))*simps(integrand(xin, m, sdcosm), xin)
#                 Gammam_Im  = -np.interp(Ri, r, sdsinm)  + 2*(m+1)*Ri**(-(2+m))*simps(integrand(xin, m, sdsinm), xin)
#                 Gamma_m_Re = -np.interp(Ri, r, sdcosm)  - 2*(-m+1)*Ri**(-(2-m))*simps(integrand(xout,-m,  sdcosm), xout)
#                 Gamma_m_Im = -np.interp(Ri, r, -sdsinm) - 2*(-m+1)*Ri**(-(2-m))*simps(integrand(xout,-m, -sdsinm), xout)
#                 Sheart_cos['+shear_cos_'+str(m)].append((Gammam_Re+Gamma_m_Re)/2)
#                 Sheart_sin['+shear_sin_'+str(m)].append(-(Gammam_Im-Gamma_m_Im)/2)
#                 Shearx_cos['xshear_cos_'+str(m)].append((Gammam_Im+Gamma_m_Im)/2)
#                 Shearx_sin['xshear_sin_'+str(m)].append((Gammam_Re-Gamma_m_Re)/2)

#         elif m==0:
#             for i, Ri in enumerate(Rproj):
#                 xdown = r[r<Ri]
#                 a1 = np.interp(Ri, r, sdcosm)
#                 a2 = simps(integrand(xdown, m, sdcosm), xdown)*2/(Ri**2)
#                 Sheart_cos['+shear_cos_'+str(m)].append(-a1 + a2)
#                 Sheart_sin['+shear_sin_'+str(m)].append(0)
#                 Shearx_cos['xshear_cos_'+str(m)].append(0)
#                 Shearx_sin['xshear_sin_'+str(m)].append(0)

#     full = {}
#     for m in m_list:
#         full['+shear_cos_'+str(m)] = Sheart_cos['+shear_cos_'+str(m)]
#         full['+shear_sin_'+str(m)] = Sheart_sin['+shear_sin_'+str(m)]
#         full['xshear_cos_'+str(m)] = Shearx_cos['xshear_cos_'+str(m)]
#         full['xshear_sin_'+str(m)] = Shearx_sin['xshear_sin_'+str(m)]
#     full['R'] = Rproj

#     return Table(full)



    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

    

# def double_integral_simps(integrand, x, y):
#     return np.trapz(simps(integrand, x, axis = 0), y)
#     #return simps(simps(integrand, x, axis = 0), y)



# def f():
#     if method=='tab':

#         n_R = 2000
#         n_phi = 100

#         sd = np.zeros([n_R, n_phi])
#         phi = np.linspace(0, 2*np.pi, n_phi)
#         r = np.logspace(np.log10(0.001), np.log10(13), n_R)
#         r[0] = 1e-9
#         Phi_, R_ = np.meshgrid(phi, r)
#         R_flat, Phi_flat = R_.flatten(), Phi_.flatten()
#         #a_eq_flat = np.sqrt((np.cos(Phi_flat-phi0)/a)**2 + (np.sin(Phi_flat-phi0)/b)**2)
#         a_term = np.cos(Phi_flat)*np.cos(phi0) + np.sin(Phi_flat)*np.sin(phi0)
#         b_term = np.cos(Phi_flat)*np.sin(phi0) - np.sin(Phi_flat)*np.cos(phi0)
#         a_eq_flat = np.sqrt((a_term/a)**2 + (b_term/b)**2)

#         r0 = .1e-2
#         sdr0 = moo.eval_surface_density(r0, z)
#         def Sd_spherical(r):
#             sd = moo.eval_surface_density(r, z)
#             mask = r < r0
#             sd[mask] = sdr0
#             return sd

#         Sd_flat = Sd_spherical(R_flat * a_eq_flat)/(a*b)
#         Sd = Sd_flat.reshape([n_R, n_phi])
#         Sdm = {'Sd'+str(m):None for m in m_list}
#         def trigo(x): return np.cos(x)
#         for m in m_list:
#             if m == 0:
#                 Sdm['Sd'+str(m)] = simps(Sd, phi, axis = 1)/(2*np.pi)
#             else:
#                 Sdm['Sd'+str(m)] = simps(Sd * trigo(m * Phi_), phi, axis = 1)/np.pi

#         esd_ = []
#         esd_2_ = []
#         esd_x_2_ = []

#         for i, Ri in enumerate(R):

#             R_inf = np.copy(R_)
#             R_inf = np.where(R_inf <= Ri, R_, 0)
#             R_sup_1 = np.where(R_ >= Ri, 1./R_, 0)
#             R_sup = np.where(R_ >= Ri, R_, 0)
#             #esd
#             if esd==True:
#                 a1 = double_integral_simps(Sd * R_inf, r, phi) * 2./( 2 * np.pi * Ri ** 2)
#                 a2 = np.interp(Ri, r, sd0)
#                 esd_.append(a1 - a2)
#             #esd_2
#             if esd_2==True:
#                 a0 = np.interp(Ri, r, sd2)
#                 alpha3 = double_integral_simps(Sd * R_inf ** 3 * trigo(2 * Phi_), r, phi) / np.pi
#                 alpha_1 = double_integral_simps(Sd * R_sup_1 * trigo(2 * Phi_), r, phi) / np.pi
#                 esd_2_.append(-a0 + alpha3 * 3/(Ri **4) + alpha_1 )
#             #esd_x_2
#             if esd_x_2==True:
#                 esd_x_2_.append( alpha3 * 3/(Ri ** 4) - alpha_1 )

#         res = []
#         if esd == True: res.append(np.array(esd_))
#         if esd_2 == True: res.append(np.array(esd_2_))
#         if esd_x_2 == True: res.append(np.array(esd_x_2_))
#         return res 

