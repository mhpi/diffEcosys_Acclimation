import torch
from models.physical.Fates_constants import *
########################################################################################################
def quadratic_min(aquad,bquad,cquad):
    #Description: Solve for the minimum root of a quadratic equation
    #Copied from: FATES , cited there as : ! Solution from Press et al (1986) Numerical Recipes: The Art of Scientific
                                          #! Computing (Cambridge University Press, Cambridge), pp. 145.


    # Inputs : aquad, bquad, cquad are the terms of a quadratic equation
    # outputs: minimum of r1 & r2 are the roots of the equation
    mask = bquad >= 0.0
    mask = mask.type(torch.uint8)
    q = -0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * mask +  \
        -0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * ( 1 - mask)

    r1   = q / aquad
    mask = q != 0.0
    mask = mask.type(torch.uint8)
    r2 = cquad / (q+prec) * mask + 1.e36 * ( 1 - mask)

    return torch.min(r1,r2)

########################################################################################################
def quadratic_max(aquad,bquad,cquad):
    # Description: Solve for the maximum root of a quadratic equation
    # Copied from: FATES , cited there as : ! Solution from Press et al (1986) Numerical Recipes: The Art of Scientific
                                          # ! Computing (Cambridge University Press, Cambridge), pp. 145.


    # Inputs : aquad, bquad, cquad are the terms of a quadratic equation
    # outputs: maximum of r1 & r2 are the roots of the equation
    mask = bquad >= 0.0
    mask = mask.type(torch.uint8)
    q = -0.5 * (bquad + torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * mask +  \
        -0.5 * (bquad - torch.sqrt(bquad*bquad - 4.0 * aquad * cquad + prec)) * ( 1 - mask)

    r1 = q / aquad
    mask = q != 0.0
    mask = mask.type(torch.uint8)
    r2 = cquad / (q+prec) * mask + 1.e36 * ( 1 - mask)

    return torch.max(r1,r2)

########################################################################################################
def ft1_f(tl,ha):
    # DESCRIPTION:photosynthesis temperature response
    # Copied from: FATES

    # Inputs :
    # tl: leaf temperature in photosynthesis temperature function (K)
    # ha: activation energy in photosynthesis temperature function (J/mol)

    # outputs: parameter scaled to leaf temperature (tl)
    return torch.exp(ha/(rgas_J_K_kmol * 1.0e-3 * (t_water_freeze_k_1atm + 25)) *
                 (1.0-(t_water_freeze_k_1atm + 25.0)/tl))

########################################################################################################

def fth25_f(hd,se):
    # Description:scaling factor for photosynthesis temperature inhibition
    # Copied from: FATES

    # Inputs :
    # hd:deactivation energy in photosynthesis temp function (J/mol)
    # se:entropy term in photosynthesis temp function (J/mol/K)

    # outputs: parameter scaled to leaf temperature (tl)
    # return 1.0 + torch.exp(torch.tensor(-hd + se * (t_water_freeze_k_1atm + 25.0)) /
    #                    (rgas_J_K_kmol * 1.0e-3 * (t_water_freeze_k_1atm+25.0)))
    return 1.0 + torch.exp((-hd + se * (t_water_freeze_k_1atm + 25.0)) /
                          (rgas_J_K_kmol * 1.0e-3 * (t_water_freeze_k_1atm+25.0)))


########################################################################################################

def fth_f(tl,hd,se,scaleFactor):
    # Description:photosynthesis temperature inhibition
    # Copied from: FATES
    # Inputs :
    # tl: leaf temperature in photosynthesis temperature function (K)
    # hd:deactivation energy in photosynthesis temp function (J/mol)
    # se:entropy term in photosynthesis temp function (J/mol/K)
    # scaleFactor  ! scaling factor for high temp inhibition (25 C = 1.0)


    return scaleFactor / (1.0 + torch.exp((-hd+se*tl) / (rgas_J_K_kmol * 1.0e-3 * tl)))

########################################################################################################
def QSat(tempk, RH):
    # Description:Computes saturation mixing ratio and the change in saturation
    # Copied from: CLM5.0

    # Parameters for derivative:water vapor
    a0 =  6.11213476;      a1 =  0.444007856     ; a2 =  0.143064234e-01 ; a3 =  0.264461437e-03
    a4 =  0.305903558e-05; a5 =  0.196237241e-07;  a6 =  0.892344772e-10 ; a7 = -0.373208410e-12
    a8 =  0.209339997e-15

    # Parameters For ice (temperature range -75C-0C)
    c0 =  6.11123516;      c1 =  0.503109514;     c2 =  0.188369801e-01; c3 =  0.420547422e-03
    c4 =  0.614396778e-05; c5 =  0.602780717e-07; c6 =  0.387940929e-09; c7 =  0.149436277e-11
    c8 =  0.262655803e-14;

    # Inputs:
    # tempk: temperature in kelvin
    # RH   : Relative humidty in fraction

    #outputs:
    # veg_esat: saturated vapor pressure at tempk (pa)
    # air_vpress: air vapor pressure (pa)
    td = torch.min(torch.tensor(100.0), torch.max(torch.tensor(-75.0), tempk - t_water_freeze_k_1atm))

    mask = td >= 0.0
    mask = mask.type(torch.uint8)
    veg_esat = (a0 + td*(a1 + td*(a2 + td*(a3 + td*(a4 + td*(a5 + td*(a6 + td*(a7 + td*a8)))))))) * mask + \
               (c0 + td*(c1 + td*(c2 + td*(c3 + td*(c4 + td*(c5 + td*(c6 + td*(c7 + td*c8)))))))) * (1 - mask)

    veg_esat = veg_esat * 100.0           # pa
    air_vpress = RH * veg_esat            # RH as fraction
    return veg_esat, air_vpress


########################################################################################################
def GetCanopyGasParameters(can_press, can_o2_partialpress, veg_tempk, air_tempk,
                            air_vpress, veg_esat, rb):

    # Description: calculates the specific Michaelis Menten Parameters (pa) for CO2 and O2, as well as
    # the CO2 compentation point.
    # Copied from: FATES

    # Inputs:
    # can_press          : Air pressure within the canopy (Pa)
    # can_o2_partialpress: Partial press of o2 in the canopy (Pa)
    # veg_tempk          : The temperature of the vegetation (K)
    # air_tempk          : Temperature of canopy air (K)
    # air_vpress         : Vapor pressure of canopy air (Pa)
    # veg_esat           : Saturated vapor pressure at veg surf (Pa)
    # rb                 : Leaf Boundary layer resistance (s/m)

    # Outputs:
    # mm_kco2   :Michaelis-Menten constant for CO2 (Pa)
    # mm_ko2    :Michaelis-Menten constant for O2 (Pa)
    # co2_cpoint:CO2 compensation point (Pa)
    # cf        :conversion factor between molar form and velocity form of conductance and resistance: [umol/m3]
    # gb_mol    :leaf boundary layer conductance (umol H2O/m**2/s)
    # ceair     :vapor pressure of air, constrained (Pa)

    kc25 = (mm_kc25_umol_per_mol / umol_per_mol) * can_press
    ko25 = (mm_ko25_mmol_per_mol / mmol_per_mol) * can_press
    sco  = 0.5 * 0.209/ (co2_cpoint_umol_per_mol / umol_per_mol)
    cp25 = 0.5 * can_o2_partialpress / sco

    mask = (veg_tempk > 150.0) & (veg_tempk < 350.0)
    mask = mask.type(torch.uint8)
    mm_kco2    = (kc25 * ft1_f(veg_tempk, kcha)) * mask + 1.0 * (1 - mask)
    mm_ko2     = (ko25 * ft1_f(veg_tempk, koha)) * mask + 1.0 * (1 - mask)
    co2_cpoint = (cp25 * ft1_f(veg_tempk, cpha)) * mask + 1.0 * (1 - mask)

    cf = can_press / (rgas_J_K_kmol * air_tempk) * umol_per_kmol
    gb_mol = (1.0 / rb) * cf
    ceair = torch.min(torch.max(air_vpress, 0.05 * veg_esat), veg_esat)

    return mm_kco2, mm_ko2, co2_cpoint, cf, gb_mol, ceair


########################################################################################################
def lmr25top_ft_extract(c3c4_path_index, vcmax25top_ft):
    # Description: calculates the canopy top leaf maint resp rate at 25C for this plant or pft (umol CO2/m**2/s)
    # for C3 plants : lmr_25top_ft = 0.015 vcmax25top_ft
    # for C4 plants : lmr_25top_ft = 0.025 vcmax25top_ft
    # From: FATES

    # Inputs:
    # c3c4_path_index: index whether pft is C3 (index = 1) or C4 (index = 0)
    # vcmax25top_ft  : canopy top maximum rate of carboxylation at 25C for this pft (umol CO2/m**2/s)

    # Outputs:
    # lmr_25top_ft   : canopy top leaf maint resp rate at 25C for this plant or pft (umol CO2/m**2/s)

    mask = c3c4_path_index == c3_path_index
    mask = mask.type(torch.uint8)
    lmr_25top_ft = (0.015 * vcmax25top_ft) * mask + (0.025 * vcmax25top_ft) * (1 - mask)
    return lmr_25top_ft

########################################################################################################
def LeafLayerMaintenanceRespiration(lmr25top_ft, nscaler,veg_tempk, c3c4_path_index):
    # Description :  Base maintenance respiration rate for plant tissues maintresp_leaf_ryan1991_baserate
    # M. Ryan, 1991. Effects of climate change on plant respiration. It rescales the canopy top leaf maint resp
    # rate at 25C to the vegetation temperature (veg_tempk)
    # From: FATES

    # Inputs:
    # lmr_25top_ft   : canopy top leaf maint resp rate at 25C for this plant or pft (umol CO2/m**2/s)
    # nscaler        : leaf nitrogen scaling coefficient (assumed here as 1)
    # veg_tempk      : vegetation temperature
    # c3c4_path_index: index whether pft is C3 (index = 1) or C4 (index = 0)

    # Outputs:
    # lmr    : Leaf Maintenance Respiration  (umol CO2/m**2/s)

    lmr25 = lmr25top_ft * nscaler  ## nscaler =1
    mask  = c3c4_path_index == 1
    mask  = mask.type(torch.uint8)

    lmr = (lmr25 * ft1_f(veg_tempk, lmrha) * fth_f(veg_tempk, lmrhd, lmrse, lmrc)) * mask + \
          (lmr25 * 2.0 ** ((veg_tempk - (t_water_freeze_k_1atm + 25.0)) / 10.0) )  * (1 - mask)

    lmr = lmr * mask + (lmr / (1.0 + torch.exp(1.3 * (veg_tempk-(t_water_freeze_k_1atm+55.0))))) * (1 - mask)

    return lmr
########################################################################################################
def LeafLayerBiophysicalRates(parsun_lsl, vcmax25top_ft, jmax25top_ft, co2_rcurve_islope25top_ft, nscaler, veg_tempk
                              ,btran, c3c4_path_index,vcmaxha, vcmaxse ):

    # Description: calculates the localized rates of several key photosynthesis rates.  By localized, we mean specific to the plant type and leaf layer,
    # which factors in leaf physiology, as well as environmental effects. This procedure should be called prior to iterative solvers, and should
    # have pre-calculated the reference rates for the pfts before this
    # From: FATES

    # Inputs:
    # parsun_lsl               :PAR absorbed in sunlit leaves for this layer
    # vcmax25top_ft            :canopy top maximum rate of carboxylation at 25C for this pft (umol CO2/m**2/s)
    # jmax25top_ft             :canopy top maximum electron transport rate at 25C for this pft (umol electrons/m**2/s)
    # co2_rcurve_islope25top_ft:initial slope of CO2 response curve (C4 plants) at 25C, canopy top, this pft
    # nscaler                  :leaf nitrogen scaling coefficient (assumed here as 1)
    # veg_tempk                :vegetation temperature
    # btran                    :transpiration wetness factor (0 to 1)
    # c3c4_path_index          :index whether pft is C3 (index = 1) or C4 (index = 0)

    # Outputs:
    # vcmax            :maximum rate of carboxylation (umol co2/m**2/s)
    # jmax             :maximum electron transport rate (umol electrons/m**2/s)
    # co2_rcurve_islope:initial slope of CO2 response curve (C4 plants)

    # Define vcmax temperature parameters
    vcmax25 = vcmax25top_ft * nscaler
    jmax25  = jmax25top_ft * nscaler
    co2_rcurve_islope25 = co2_rcurve_islope25top_ft * nscaler

    vcmaxhd = torch.zeros_like(torch.clone(c3c4_path_index)[0]) + vcmaxhd_FATES

    # Define jmax temperature parameters
    jmaxhd = torch.zeros_like(torch.clone(c3c4_path_index)[0]) + jmaxhd_FATES
    jmaxha = torch.zeros_like(torch.clone(c3c4_path_index)[0]) + jmaxha_FATES
    jmaxse = torch.zeros_like(torch.clone(c3c4_path_index)[0]) + jmaxse_FATES

    vcmaxc = fth25_f(vcmaxhd, vcmaxse)
    jmaxc = fth25_f(jmaxhd, jmaxse)

    vcmax = vcmax25 * ft1_f(veg_tempk, vcmaxha) * fth_f(veg_tempk, vcmaxhd, vcmaxse, vcmaxc)
    jmax  = jmax25  * ft1_f(veg_tempk, jmaxha)  * fth_f(veg_tempk, jmaxhd, jmaxse, jmaxc)
    co2_rcurve_islope = co2_rcurve_islope25 * 2.0 ** ((veg_tempk - (t_water_freeze_k_1atm + 25.0)) / 10.0)

    mask = c3c4_path_index != 1
    mask = mask.type(torch.uint8)

    vcmax = vcmax * (1 - mask) + (vcmax25 * 2.0 ** ((veg_tempk - (t_water_freeze_k_1atm + 25.0)) / 10.0)) * mask
    vcmax = vcmax * (1 - mask) + (vcmax / (1.0 + torch.exp(0.2 * ((t_water_freeze_k_1atm+15.0) - veg_tempk)))) * mask
    vcmax = vcmax * (1 - mask) + (vcmax / (1.0 + torch.exp(0.3 * (veg_tempk-(t_water_freeze_k_1atm+40.0))))) * mask

    mask = (parsun_lsl <= 0.0)
    mask = mask.type(torch.uint8)

    vcmax = 0.0 * mask + vcmax * (1 - mask)
    jmax  = 0.0 * mask + jmax  * (1 - mask)
    co2_rcurve_islope = 0.0 * mask + co2_rcurve_islope * (1 - mask)

    vcmax = vcmax * btran

    return vcmax, jmax, co2_rcurve_islope
########################################################################################################






