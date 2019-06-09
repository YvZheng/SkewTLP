# -*- coding: utf-8 -*-
"""
大气物理与大气动力学函数
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

Rs_da = 287.05  # Specific gas const for dry air, J kg^{-1} K^{-1}
Rs_v = 461.51  # Specific gas const for water vapour, J kg^{-1} K^{-1}
Cp_da = 1004.6  # Specific heat at constant pressure for dry air
Cv_da = 719.  # Specific heat at constant volume for dry air
Cp_v = 1870.  # Specific heat at constant pressure for water vapour
Cv_v = 1410.  # Specific heat at constant volume for water vapour
Cp_lw = 4218  # Specific heat at constant pressure for liquid water
Epsilon = 0.622  # Epsilon=Rs_da/Rs_v; The ratio of the gas constants
degCtoK = 273.15  # Temperature offset between K and C (deg C)
rho_w = 1000.  # Liquid Water density kg m^{-3}
grav = 9.80665  # Gravity, m s^{-2}
Lv = 2.5e6  # Latent Heat of vaporisation
boltzmann = 5.67e-8  # Stefan-Boltzmann constant
mv = 18.0153e-3  # Mean molar mass of water vapor(kg/mol)
m_a = 28.9644e-3  # Mean molar mass of air(kg/mol)
Rstar_a = 8.31432  # Universal gas constant for air (N m /(mol K))


def barometric_equation(presb_pa, tempb_k, deltah_m, Gamma=-0.0065):
    """正压大气公式由底层气压（Pa）、底层温度（K）、距离底层的距离（m），
    计算该层高度的气压（Pa）
    The barometric equation models the change in pressure with
    height in the atmosphere.

    INPUTS:
    :param presb_k (pa):     The base pressure
    :param tempb_k (K):      The base temperature
    :param deltah_m (m):     The height differential between the base height and the
                      desired height
    :param Gamma [=-0.0065]: The atmospheric lapse rate

    OUTPUTS
    :param pres (pa):        Pressure at the requested level

    REFERENCE:
    http://en.wikipedia.org/wiki/Barometric_formula
    """

    return presb_pa * \
           (tempb_k / (tempb_k + Gamma * deltah_m)) ** (grav * m_a / (Rstar_a * Gamma))


def barometric_equation_inv(heightb_m, tempb_k, presb_pa,
                            prest_pa, Gamma=-0.0065):
    """
    正压大气公式由底层高度（m）、底层温度（K）、底层的气压（Pa）、目标层的气压（Pa），
    计算目标层的高度（m）
    The barometric equation models the change in pressure with height in
    the atmosphere. This function returns altitude given
    initial pressure and base altitude, and pressure change.

    INPUTS:
    :param heightb_m (m):
    :param presb_pa (pa):    The base pressure
    :param tempb_k (K)  :    The base temperature
    :param deltap_pa (m):    The pressure differential between the base height and the
                      desired height

    :param Gamma [=-0.0065]: The atmospheric lapse rate

    OUTPUTS
    :param heightt_m

    REFERENCE:
    http://en.wikipedia.org/wiki/Barometric_formula
    """

    return heightb_m + \
           tempb_k * ((presb_pa / prest_pa) ** (Rstar_a * Gamma / (grav * m_a)) - 1) / Gamma


def Theta(tempk, pres, pref=100000.):
    """
    由当前的温度（K），气压（Pa），计算位温（K）
    Potential Temperature

    INPUTS:
    :param tempk (K)
    :param pres (Pa)
    :param pref: Reference pressure (default 100000 Pa)

    OUTPUTS:
    :param Theta (K)

    Source: Wikipedia
    Prints a warning if a pressure value below 2000 Pa input, to ensure
    that the units were input correctly.
    """
    return tempk * (pref / pres) ** (Rs_da / Cp_da)


def TempK(theta, pres, pref=100000.):
    """Inverts Theta function.
    根据位温（K）以及当前气压（Pa）计算当前温度（K）
    INPUTS:
    :param theta (K)
    :param pres (Pa)
    :param pref: Reference pressure (default 100000 Pa)

    OUTPUTS:
    :param tempK (K)

    """
    return theta * (pres / pref) ** (Rs_da / Cp_da)


def ThetaE(tempk, pres, e):
    """
    根据当前温度（K）、当前气压（Pa）、水汽压（e）计算相当位温（K）
    Calculate Equivalent Potential Temperature
        for lowest model level (or surface)

    INPUTS:
    :param tempk:      Temperature [K]
    :param pres:       Pressure [Pa]
    :param e:          Water vapour partial pressure [Pa]

    OUTPUTS:
    :param theta_e:    equivalent potential temperature

    References:
    Eq. (9.40) from Holton (2004)
    Eq. (22) from Bolton (1980)
    Michael P. Byrne and Paul A. O'Gorman (2013), 'Land-Ocean Warming
    Contrast over a Wide Range of Climates: Convective Quasi-Equilibrium
    Theory and Idealized Simulations', J. Climate """

    # tempc
    tempc = tempk - degCtoK
    # Calculate theta
    theta = Theta(tempk, pres)

    # T_lcl formula needs RH
    es = VaporPressure(tempk)
    RH = 100. * e / es

    # theta_e needs q (water vapour mixing ratio)
    qv = MixRatio(e, pres)
    # Calculate the temp at the Lifting Condensation Level
    T_lcl = ((tempk - 55) * 2840 / (2840 - (np.log(RH / 100) * (tempk - 55)))) + 55
    # print "T_lcl :%.3f"%T_lcl
    # DEBUG STUFF ####
    theta_l = tempk * \
              (100000. / (pres - e)) ** (Rs_da / Cp_da) * (tempk / T_lcl) ** (0.28 * qv)
    # print "theta_L: %.3f"%theta_l
    # Calculate ThetaE
    theta_e = theta_l * np.exp((Lv * qv) / (Cp_da * T_lcl))
    return theta_e


def ThetaE_Bolton(tempk, pres, e, pref=100000.):
    """
    根据当前温度（K）、当前气压（Pa）、水汽压（e）计算相当位温（K）
    Theta_E following Bolton (1980)
    INPUTS:
    :param tempk:      Temperature [K]
    :param pres:       Pressure [Pa]
    :param e:          Water vapour partial pressure [Pa]
    OUTPUTS:
    :param theta_e:    equivalent potential temperature

    See http://en.wikipedia.org/wiki/Equivalent_potential_temperature
    """

    # Preliminary:
    T = tempk
    qv = MixRatio(e, pres)
    Td = DewPoint(e)
    kappa_d = Rs_da / Cp_da

    # Calculate TL (temp [K] at LCL):
    TL = 56 + ((Td - 56.) ** -1 + (np.log(T / Td) / 800.)) ** (-1)
    # print "TL: %.3f"%TL
    # Calculate Theta_L:
    thetaL = T * (pref / (pres - e)) ** kappa_d * (T / TL) ** (0.28 * qv)
    # print "theta_L: %.3f"%thetaL
    # put it all together to get ThetaE
    thetaE = thetaL * np.exp((3036. / TL - 0.78) * qv * (1 + 0.448 * qv))

    return thetaE


def ThetaV(tempk, pres, e):
    """
    根据当前温度（K）、当前气压（Pa）、水汽压（e）计算虚位温（K）
    Virtual Potential Temperature

    INPUTS
    :param tempk (K)
    :param pres (Pa)
    :param e: Water vapour pressure (Pa)

    OUTPUTS
    :param theta_v    : Virtual potential temperature
    """

    mixr = MixRatio(e, pres)
    theta = Theta(tempk, pres)
    return theta * (1 + mixr / Epsilon) / (1 + mixr)


def GammaW(tempk, pres):
    """
    根据温度（K），气压（Pa）计算湿绝热递减率（gamma）
    Function to calculate the moist adiabatic lapse rate (deg C/Pa) based
    on the environmental temperature and pressure.

    INPUTS:
    :param tempk (K)
    :param pres (Pa)

    RETURNS:
    :param GammaW: The moist adiabatic lapse rate (Deg C/Pa)
    REFERENCE:
    http://glossary.ametsoc.org/wiki/Moist-adiabatic_lapse_rate
    (Note that I multiply by 1/(grav*rho) to give MALR in deg/Pa)

    """

    tempc = tempk - degCtoK
    es = VaporPressure(tempk)
    ws = MixRatio(es, pres)

    # tempv=VirtualTempFromMixR(tempk,ws)
    tempv = VirtualTemp(tempk, pres, es)
    latent = Latentk(tempk)

    Rho = pres / (Rs_da * tempv)

    # This is the previous implementation:
    # A=1.0+latent*ws/(Rs_da*tempk)
    # B=1.0+Epsilon*latent*latent*ws/(Cp_da*Rs_da*tempk*tempk)
    # Gamma=(A/B)/(Cp_da*Rho)

    # This is algebraically identical but a little clearer:
    A = -1. * (1.0 + latent * ws / (Rs_da * tempk))
    B = Rho * (Cp_da + Epsilon * latent * latent * ws / (Rs_da * tempk * tempk))
    Gamma = A / B
    return Gamma


def DensHumid(tempk, pres, e):
    """
    由气温（K）,气压（Pa），水汽压（Pa），计算湿空气密度（kg/m^3）
    Density of moist air.
    This is a bit more explicit and less confusing than the method below.

    INPUTS:
    :param tempk: Temperature (K)
    :param pres: static pressure (Pa)
    :param e:  Water vapour partial pressure [Pa]

    OUTPUTS:
    :param rho_air (kg/m^3)

    SOURCE: http://en.wikipedia.org/wiki/Density_of_air
    """

    pres_da = pres - e
    rho_da = pres_da / (Rs_da * tempk)
    rho_wv = e / (Rs_v * tempk)

    return rho_da + rho_wv


def Density(tempk, pres, mixr):
    """
    由气温（K）,气压（Pa），水汽混合比（kg/kg），计算湿空气密度（kg/m^3）
    Density of moist air

    INPUTS:
    :param tempk: Temperature (K)
    :param pres: static pressure (Pa)
    :param mixr: mixing ratio (kg/kg)

    OUTPUTS:
    :param rho_air (kg/m^3)
    """

    virtualT = VirtualTempFromMixR(tempk, mixr)
    return pres / (Rs_da * virtualT)


def VirtualTemp(tempk, pres, e):
    """
    根据气温（K），气压（Pa），水汽压（Pa）计算虚温（K）
    Virtual Temperature

    INPUTS:
    :param tempk: Temperature (K)
    :param e: vapour pressure (Pa)
    :param pres: static pressure (Pa)

    OUTPUTS:
    :param tempv: Virtual temperature (K)

    SOURCE: hmmmm (Wikipedia)."""

    tempvk = tempk / (1 - (e / pres) * (1 - Epsilon))
    return tempvk


def VirtualTempFromMixR(tempk, mixr):
    """
    根据气温（K），气压（Pa），水汽混合比（kg/kg），计算虚温（K）
    Virtual Temperature

    INPUTS:
    :param tempk: Temperature (K)
    :param mixr: Mixing Ratio (kg/kg)

    OUTPUTS:
    :param tempv: Virtual temperature (K)

    SOURCE: hmmmm (Wikipedia). This is an approximation
    based on a m
    """
    return tempk * (1.0 + 0.6 * mixr)


def Latentk(tempk):
    """
    根据温度(k)计算潜热能（J/kg）
    Latent heat of condensation (vapourisation)

    INPUTS:
    :param tempk (k)

    OUTPUTS:
    :param L_w (J/kg)

    SOURCE:
    http://en.wikipedia.org/wiki/Latent_heat#Latent_heat_for_condensation_of_water
    """
    tempc = tempk - degCtoK
    return 1000 * (2500.8 - 2.36 * tempc + 0.0016 * tempc ** 2 - 0.00006 * tempc ** 3)


def VaporPressure(tempk, phase="liquid"):
    """
    由计算气温（露点）tempk(k) 计算饱和水汽压（水汽压）（Pa）
    Water vapor pressure over liquid water or ice.

    INPUTS:
    :param tempk: (K) OR dwpt (K), if SATURATION vapour pressure is desired.
    :param phase: ['liquid'],'ice'. If 'liquid', do simple dew point. If 'ice',
    return saturation vapour pressure as follows:

    Tc>=0: es = es_liquid
    Tc <0: es = es_ice
    RETURNS:
    :param e_sat / e (Pa)

    SOURCE: http://cires.colorado.edu/~voemel/vp.html (#2:
    CIMO guide (WMO 2008), modified to return values in Pa)

    This formulation is chosen because of its appealing simplicity,
    but it performs very well with respect to the reference forms
    at temperatures above -40 C. At some point I'll implement Goff-Gratch
    (from the same resource).
    Notes
    -----
    Instead of temperature, dewpoint may be used in order to calculate
    the actual (ambient) water vapor (partial) pressure.

    The formula used is that from [Bolton1980]_ for T in degrees Celsius:

    .. math:: 6.112 e^\frac{17.67T}{T + 243.5}
    """
    tempc = tempk - degCtoK
    over_liquid = 6.112 * np.exp(17.67 * tempc / (tempc + 243.12)) * 100.
    over_ice = 6.112 * np.exp(22.46 * tempc / (tempc + 272.62)) * 100.

    if phase == "liquid":
        return over_liquid
    elif phase == "ice":
        return np.where(tempc < 0, over_ice, over_liquid)
    else:
        raise NotImplementedError


def MixRatio(e, pres):
    """
    由水汽压（Pa）和气压（Pa）计算混合比（kg/kg）
    Mixing ratio of water vapour
    INPUTS
    :param e (Pa) Water vapor pressure
    :param pres (Pa) Ambient pressure

    RETURNS
    :param mixr (kg kg^-1) Water vapor mixing ratio`
    """

    return Epsilon * e / (pres - e)


def MixR2VaporPress(mixr, pres):
    """
    由混合比mixr(kg/kg)和气压（Pa）计算水汽压（Pa）
    Return Vapor Pressure given Mixing Ratio and Pressure
    INPUTS
    :param mixr (kg kg^-1) Water vapor mixing ratio`
    :param pres (Pa) Ambient pressure

    RETURNS
    :param e (Pa) Water vapor pressure
    """
    return mixr * pres / (Epsilon + mixr)


def DewPoint(e):
    """
    由水汽压(Pa)计算露点温度（K）
    Use Bolton's (1980, MWR, p1047) formulae to find tdew.
    INPUTS:
    :param e (Pa) Water Vapor Pressure
    OUTPUTS:
    :param Td (K)
      """
    ln_ratio = np.log(e / 611.2)
    dwptk = ((17.67 - ln_ratio) * degCtoK + 243.5 * ln_ratio) / (17.67 - ln_ratio)
    return dwptk


def WetBulb(tempk, rh):
    """
    由温度（K）和相对湿度（%）计算湿球温度（K）
    Stull (2011): Wet-Bulb Temperature from Relative Humidity and Air
    Temperature.
    INPUTS:
    :param tempk (K)
    :param rh ### (%)
    OUTPUTS:
    :param tempwb (K)
    """
    tempc = tempk - degCtoK
    TwC = tempc * np.arctan(0.151977 * (rh + 8.313659) ** 0.5) + \
          np.arctan(tempc + rh) - np.arctan(rh - 1.676331) + \
          0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - \
          4.686035
    return TwC + degCtoK


def sat_mixing_ratio(pres, tempk):
    """
    由气压P（Pa）、温度（K）计算饱和混合比(kg/kg)
    Calculate the saturation mixing ratio in kg/kg based on pressure in
       millibars and temperature in Celsius. The calculation is based on the
       following expression:
                               eps*e_s(T)
                        w_s = ------------
                                P-e_s(T)
    inputs：
    :param tempk(k),
    :param  p(Pa)
    return:
    :param mixr_s(kg/kg)
    """
    e_s = VaporPressure(tempk)
    mixr_s = (Epsilon * e_s) / (pres - e_s)
    return mixr_s


def RH(tempk, pres, mixr):
    """
    由气温（K）、气压（Pa）、比湿（kg/kg）计算相对湿度rh(%)
    Calculate the Relative Humidity in percent from the temperature in
       Celsius, the Pressure in Pa, and the mixing ratio in kg/kg.
    inputs:
    :param tempk(k),
    :param Pres(Pa)
    outputs:
    :param rh(%)
    """
    mixr_s = sat_mixing_ratio(pres, tempk)
    rh = 100.0 * mixr / mixr_s
    return rh


def T_LCL(tempk, rh):
    """
    根据抬升高度的温度（K）、相对湿度（%）计算抬升凝结高度（LCL）的温度（K）
    Calculate the temperature in Kelvin at the Lifting Condensation Level
    (LCL) based on the temperature in Kelvin and the relative humidity in
    percent. Formulation from eq. 22 of Bolton (1980).
    inputs:
    :param RH(%)
    :param tempk(K)
    :param output:
    outputs：
    :param T_lcl(K)
    """
    fracBot1 = 1.0 / (tempk - 55.0)
    fracBot2 = np.log(rh / 100.0) / 2840.0
    T_lcl = 55.0 + (1.0 / (fracBot1 - fracBot2))
    return T_lcl


def pseudoeq_potential_T(tempk, pres, mixr, P_0=100000.0):
    """
    根据气温（K），气压（Pa），比湿（kg/kg）计算假相当位温（K）
    Calculate the pseudoequivalent potential temperature in Kelvin given
       temperature in Kelvin, pressure in Pa, and mixing ratio in
       kg/kg. Formulation from eq. 43 in Bolton (1980) where r is in g/kg
       and so we must convert the input w to g/kg. return thetaEP (Kelvin)
    INPUTs:
    :param tempk(K)
    :param pres(Pa)
    :param mixr(kg/kg)
    outputs:
    :param thetaEP (K)
    """
    r = mixr * 1000.0  # kg/kg --> g/kg
    term1exponent = 0.2854 * (1.0 - 0.28 * 0.001 * r)
    term1 = tempk * np.power((P_0 / pres), term1exponent)
    rh = RH(tempk, pres, mixr)
    T_lcl = T_LCL(tempk, rh)
    term2part1 = (3.376 / T_lcl) - 0.00254
    term2part2 = r * (1.0 + 0.81 * 0.001 * r)
    term2 = np.exp(term2part1 * term2part2)
    thetaEP = term1 * term2
    return thetaEP


def RHFromDwpt(tempk, dwptk):
    """
    根据温度(K)和露点温度(K)计算相对湿度(%)
    inputs:
    :param tempk（K）
    :param dwptk（K）
    outputs:
    :param rh (%)
    """
    e = VaporPressure(dwptk)
    es = VaporPressure(tempk)
    return e / es * 100


def WetBulbFromDwpt(tempk, dwptk):
    """
    根据气温(K)和露点温度(K)计算湿球温度(K)
    inputs:
    :param tempk（K）
    :param dwptk（K）
    outputs:
    :param wetbulk(K)
    """
    RH = RHFromDwpt(tempk, dwptk)
    return WetBulb(tempk, RH)


def SpecificVolume(pres, tempk, dwptk):
    """
    根据pres(Pa)气压, tempk气温(k), dwptk露点温度(K) ,计算比容1/rho
    inputs:
    :param pres(Pa)
    :param tempk（K）
    :param dwptk（K）
    outputs:
    :param specificVolume(1/rho)
    """
    e = VaporPressure(dwptk)
    rho = DensHumid(tempk, pres, e)
    return 1. / rho


def PseudoeqPotentialTempkFromDwptk(tempk, dwptk, pres):
    """
    根据气温tempk(K)、露点温度dwptk(K)、气压（Pa），计算假相当位温（K）
    inputs:
    :param tempk（K）
    :param dwptk（K）
    :param pres(Pa)
    outputs:
    :param pseudoeq_potential_T(K)
    """
    e = VaporPressure(dwptk)
    mixr = MixRatio(e, pres)
    return pseudoeq_potential_T(tempk, pres, mixr)


def DwptkFromRH(rh, tempk):
    """根据相对湿度(%)、气温(K),计算露点温度(K)
    inputs:
    :param rh （%）
    :param tempk（K）
    outputs:
    :param Dwptk(K)
    """
    e_s = VaporPressure(tempk)
    e = e_s * rh / 100.
    return DewPoint(e)
