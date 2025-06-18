rgas_J_K_kmol         = 8314.4598 ;#universal gas constant [J/K/kmol]
t_water_freeze_k_1atm = 273.15    ;#freezing point of water at triple point (K)
umol_per_mol          = 1.0e6     ;#Conversion factor: micromoles per mole
mmol_per_mol          = 1000.0    ;#Conversion factor: milimoles per mole
umol_per_kmol         = 1.0e9     ;#Conversion factor: micromoles per kmole
rgas_J_K_mol          = 8.3144598 ;#universal gas constant [J/K/mol]
nearzero              = 1.0e-30   ;#FATES use this in place of logical comparisons between reals with zero, as the chances are their precisions are preventing perfect zero in comparison
c3_path_index         = 1         ;#Constants used to define C3 versus C4 photosynth pathways
itrue                 = 1         ;#Integer equivalent of true
medlyn_model          = 2         ;#Constants used to define conductance models
ballberry_model       = 1         ;#Constants used to define conductance models
molar_mass_water      = 18.0      ;#Molar mass of water (g/mol)

#params
mm_kc25_umol_per_mol = 404.9      ;#Michaelis-Menten constant for CO2 at 25C
mm_ko25_mmol_per_mol = 278.4      ;#Michaelis-Menten constant for O2 at 25C
co2_cpoint_umol_per_mol = 42.75   ;#CO2 compensation point at 25C
kcha                 = 79430.0    ;#activation energy for kc (J/mol)
koha                 = 36380.0    ;#activation energy for ko (J/mol)
cpha                 = 37830.0    ;#activation energy for cp (J/mol)

lmrha                = 46390.0    ;#activation energy for lmr (J/mol)
lmrhd                = 150650.0   ;#deactivation energy for lmr (J/mol)
lmrse                = 490.0      ;#entropy term for lmr (J/mol/K)
lmrc                 = 1.15912391 ;#scaling factor for high


quant_eff            = 0.05           ;#quantum efficiency, used only for C4 (mol CO2 / mol photons)
vcmaxha_FATES        = 65330          ;#activation energy for vcmax (J/mol)
vcmaxhd_FATES        = 149250         ;#deactivation energy for vcmax (J/mol)
vcmaxse_FATES        = 485            ;#entropy term for vcmax (J/mol/k)
jmaxha_FATES         = 43540          ;#activation energy for jmax (J/mol)
jmaxhd_FATES         = 152040         ;#deactivation energy for jmax (J/mol)
jmaxse_FATES         = 495            ;#entropy term for jmax (J/mol/k)
prec                 = 1e-8           ;#Avoid zeros to avoid Nans
stomatal_intercept   = [40000, 10000] ;#Unstressed minimum stomatal conductance 10000 for C3 and 40000 for C4 (umol m-2 s-1)

fnps                 = 0.15       ;#Fraction of light absorbed by non-photosynthetic pigments
theta_psii           = 0.7        ;#empirical curvature parameter for electron transport rate
theta_cj_c3          = 0.999      ;#empirical curvature parameters for ac, aj photosynthesis co-limitation, c3
theta_cj_c4          = 0.999      ;#empirical curvature parameters for ac, aj photosynthesis co-limitation, c4
theta_ip             = 0.999      ;#empirical curvature parameter for ap photosynthesis co-limitation
h2o_co2_bl_diffuse_ratio    = 1.4 ;#Ratio of H2O/CO2 gass diffusion in the leaf boundary layer (approximate)
h2o_co2_stoma_diffuse_ratio = 1.6 ;#Ratio of H2O/CO2 gas diffusion in stomatal airspace (approximate)
nscaler             =1.0          ;#leaf nitrogen scaling coefficient (assumed here as 1)
f_sun_lsl           =1.0          ;#fraction of sunlit leaves (assumed = 1)
init_a2l_co2_c3     = 0.7         ;#First guess on ratio between intercellular co2 and the atmosphere (C3)
init_a2l_co2_c4     = 0.4         ;#First guess on ratio between intercellular co2 and the atmosphere (C4)
rsmax0              = 2.0e8       ;#maximum stomatal resistance [s/m] (used across several procedures
molar_mass_ratio_vapdry = 0.622   ;#Approximate molar mass of water vapor to dry air

FPAR = 0.85                       ;#Absorption efficiency of active radiation
#vcmax25top         = [50, 65, 39, 62, 41, 58, 62, 54, 54, 78, 78, 78];
