import math
import numpy as np
import scipy as sp
from scipy.interpolate import CubicSpline



#constants
h_eff = 86.25 #effective entropic degree of freedom
g_st = 86.25**(1/2) #effective energetic degree of freedom
m_pl = 1.2 * 1e19 #planck mass, GeV
m_h = 125       #higgs mass, GeV
mu = np.sqrt(np.pi / 45) * m_pl * g_st**(1/2) #scaling constant
v0 = 246   #vacuum expectation value, GeV

#Q (GeV)
Q = np.array([80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 220.0, 240.0, 260.0, 280.0, 300.0])
#Gamma/width (MeV)
Gamma_h = np.array([1.99, 2.22, 2.48, 2.85, 3.51, 4.91, 8.17, 17.3, 83.1, 380, 631, 1040, 1430, 2310, 3400, 4760, 6430, 8430]) / 1000
#in GeV
spline_Gamma_h = CubicSpline(Q, np.log10(Gamma_h), extrapolate=True) 
def convert(ov):
    #Converts from units of cm^3/s to GeV
    return ov / (1.17 * 1e-17)

def gamma_h(m):
    #Returns interpolated/extrapolated predictions for the higgs boson decay at a given mass

    if m > 300:
        return Gamma_h[-1] + 0.1 * (m - 300) 
    return 10**spline_Gamma_h(m) 
    

def integrand_sv_eff_t(t, x, m, lambda_h):
    # x is Effective temperature of the universe (x = particle mass / temperature)
    # m is Mass of the particle
    # lambda_h is Dimensionless coupling that describes the interaction strength between DM and higgs boson.
    # Get out sv_eff which is the integrand at a specific value of x
    
    s = 1 / t   
    sqrts = np.sqrt(s)  #define for simplicity
    Dhs = 1 / ((s - m_h**2)**2 + m_h**2 * gamma_h(m_h)**2) 
    ovcms = 2 * (lambda_h * v0)**2 / sqrts * Dhs * gamma_h(sqrts)
    sv_eff =  x * s * np.sqrt(s - 4 * m**2) * sp.kn(1, x * sqrts / m) * ovcms / (16 * m**5 * sp.kn(2, x)**2 * t**2) #integrand in eq 10.
    return sv_eff

def O_effV(x, m, lambda_h):
    # x is Effective temperature of the universe (x = particle mass / temperature)
    # m is Mass of the particle
    # lambda_h is Dimensionless coupling that describes the interaction strength between DM and higgs boson.
    # Get out integral which is the the effective thermally averaged cross section for the particle at the universe epoch defined from x

    smin = 4.0 * pow(m, 2) 
    smax = 20.0 * pow(m, 2) 
    integral, err = integ.quad(integrand_sv_eff_t, 1.0/smax, 1.0/smin, args=(x, m, lambda_h), epsabs=0, epsrel=1.0e-3) #integrate
    return integral

def Y_eq(x, spin):
    # get out the equilibrium abundance of a particle given by Maxwell-Boltzmann approximation
    # where x is effective temperature of the universe (x = particle mass / temperature)
    # and spin is Particle spin

    return (45 / (4 * np.pi**4)) * (x**2 / h_eff) * (2 * spin + 1) * sp.kn(2, x)
