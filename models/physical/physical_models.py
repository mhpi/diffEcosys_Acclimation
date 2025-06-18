import torch.nn
import numpy as np
from models.solver.solver import *

#===================================================================================================================

class leaflayerphotosynthesis_model(torch.nn.Module):

    def __init__(self,flag       = ""):
        super(leaflayerphotosynthesis_model, self).__init__()
        self.flag    = flag

        if self.flag == "global":
            self.at_leaf = False
        else:
            self.at_leaf = True

    def set_forc_attrs(self, forcing_attrs):
        self.forcing_attrs  = forcing_attrs

    def set_params(self, params):
        self.params= params


    def pre_fates(self):

        # Description: This function computes some physical variables and parameters required
        # for forwarding the physical model in the differentiable framework


        # Outputs:
        # model_inputs dictionary with some added variables or parameters as:
            # vcmax25top_ft             : canopy top maximum rate of carboxylation at 25C based on pft (umol CO2/m**2/s)
            # jmax25top_ft              : canopy top maximum electron transport rate at 25C based on pft (umol electrons/m**2/s)
            # co2_rcurve_islope25top_ft : initial slope of CO2 response curve (C4 plants) at 25C, canopy top, based on pft
            # lmr25top_ft               : canopy top leaf maint resp rate at 25C based on pft (umol CO2/m**2/s)
            # vcmaxha                   : activation energy for vcmax (J/mol)
            # vcmaxse                   : entropy term for vcmax (J/mol/k)
            # vcmax                     : maximum rate of carboxylation (umol co2/m**2/s)
            # jmax                      : maximum electron transport rate (umol electrons/m**2/s)
            # co2_rcurve_islope         : initial slope of CO2 response curve (C4 plants)
            # medlyn_slope              : Slope for Medlyn stomatal conductance model method, the unit is KPa^0.5
            # stomatal_intercept_btran  : Unstressed minimum stomatal conductance 10000 for C3 and 40000 for C4 (umol m-2 s-1)
            # btran                     : transpiration wetness factor (0 to 1)
            # je                        : electron transport rate (umol electrons/m**2/s)
            # stomatal_intercept_btran_Med: minimum stomatal conductance with water limitations applied (i.e. multiplied by btran) (umol m-2 s-1)

        model_inputs               = self.forcing_attrs.copy()
        vcmax25top_ft              = self.params['Vcmax25'] if self.params['Vcmax25'] != None else model_inputs['Vcmax25_CLM45']
        btran                      = self.params['btran'] if self.params['btran'] != None else model_inputs['btran_CLM45']


        jmax25top_ft              = 1.67 * vcmax25top_ft
        co2_rcurve_islope25top_ft = 20000 * vcmax25top_ft
        lmr25top_ft               = lmr25top_ft_extract(model_inputs['c3c4_path_index'], vcmax25top_ft)

        lmr                       = LeafLayerMaintenanceRespiration(lmr25top_ft, nscaler, model_inputs['veg_tempk'], model_inputs['c3c4_path_index'])

        vcmaxha = self.params['vcmaxha'] if self.params['vcmaxha'] != None else torch.zeros_like(torch.clone(model_inputs['c3c4_path_index'])[0]) + vcmaxha_FATES
        vcmaxse = self.params['vcmaxse'] if self.params['vcmaxse'] != None else torch.zeros_like(torch.clone(model_inputs['c3c4_path_index'])[0]) + vcmaxse_FATES

        vcmax, jmax, co2_rcurve_islope = LeafLayerBiophysicalRates(parsun_lsl       =model_inputs['parsun_lsl'],
                                                                   vcmax25top_ft    =vcmax25top_ft,
                                                                   jmax25top_ft     =jmax25top_ft,
                                                                   co2_rcurve_islope25top_ft = co2_rcurve_islope25top_ft,
                                                                   nscaler          =nscaler,
                                                                   veg_tempk        =model_inputs['veg_tempk'],
                                                                   btran            =btran,
                                                                   c3c4_path_index  =model_inputs['c3c4_path_index'],
                                                                   vcmaxha          =vcmaxha,
                                                                   vcmaxse          =vcmaxse
                                                                   )

        stomatal_intercept_btran     = self.params['g0'] if self.params['g0'] != None else model_inputs['stomatal_intercept_btran']
        medlyn_slope                 = self.params['g1'] if self.params['g1'] != None else model_inputs['medlyn_slope']
        stomatal_intercept_btran_Med = torch.max(model_inputs['cf'] / rsmax0, stomatal_intercept_btran * btran)
        je = quadratic_min(theta_psii, - (model_inputs['qabs'] + jmax), model_inputs['qabs'] * jmax)

        model_inputs['lmr']              = lmr
        model_inputs['vcmax']            = vcmax
        model_inputs['jmax']             = jmax
        model_inputs['co2_rcurve_islope']= co2_rcurve_islope
        model_inputs['je']               = je
        model_inputs['stomatal_intercept_btran_Med'] = stomatal_intercept_btran_Med
        model_inputs['medlyn_slope']                 = medlyn_slope

        return model_inputs
    def get_guess(self, model_inputs):
        # Description: A function to get the initial guess for the intercellular leaf CO2 pressure (Ci) which is mainly
        # dependent on the classification of C3 and C4 plants.

        # Inputs:
        # c3c4_path_index: Index for which photosynthetic pathway is active.  C4 = 0,  C3 = 1
        # can_co2_ppress : Partial pressure of CO2 NEAR the leaf surface (Pa)

        # Outputs:
        # ci: intercellular leaf CO2 pressure (Ci) in (Pa)
        mask = model_inputs['c3c4_path_index'] == c3_path_index
        mask = mask.type(torch.uint8)
        ci   = (init_a2l_co2_c3 * model_inputs['co2_ppress']) * mask + (init_a2l_co2_c4 * model_inputs['co2_ppress']) * (1 - mask)

        return ci

    def forward_model(self, model_inputs, x):
        # Description: A function representing the leaf layer photosynthesis module from FATES module
        # https: // doi.org / 10.5281 / zenodo.3825474

        # First: compute the gross photosynthesis rate (agross):

        ac = (model_inputs['vcmax'] * torch.clamp(x - model_inputs['co2_cpoint'], min=0.0) / (x + model_inputs['mm_kco2'] * (1.0 + model_inputs['can_o2_ppress'] / model_inputs['mm_ko2']))) * model_inputs['c3c4_path_index'] \
             + model_inputs['vcmax']  * (1 - model_inputs['c3c4_path_index'])

        aj = (model_inputs['je'] * torch.clamp(x - model_inputs['co2_cpoint'], min=0.0) / (4.0 * x + 8.0 * model_inputs['co2_cpoint'])) * model_inputs['c3c4_path_index'] + \
             (quant_eff * model_inputs['parsun_lsl'] * 4.6) * (1 - model_inputs['c3c4_path_index'])

        ap = model_inputs['co2_rcurve_islope'] * torch.clamp(x, min=0.0) / model_inputs['can_press']

        ai = 0.0 * model_inputs['c3c4_path_index'] + quadratic_min(theta_cj_c4, -(ac + aj), ac * aj) * (1 - model_inputs['c3c4_path_index'])

        agross = quadratic_min(theta_cj_c3, -(ac + aj), ac * aj) * model_inputs['c3c4_path_index'] + \
                 quadratic_min(theta_ip, - (ai + ap), ai * ap) * (1 - model_inputs['c3c4_path_index'])

        # Second: compute the net photosynthesis rate (anet):
        anet = agross - model_inputs['lmr']

        # Third:  correct anet for LAI > 0.0 and parsun_lsl <=0.0
        mask = model_inputs['LAI'] > 0.0
        mask = mask.type(torch.uint8)
        anet = mask * anet + (1 - mask) * 0.0

        mask = model_inputs['parsun_lsl'] <= 0.0
        mask = mask.type(torch.uint8)
        anet = mask * -model_inputs['lmr'] + (1 - mask) * anet

        # Fourth: Stomatal Conductance computation
        a_gs = anet

        if self.at_leaf:
            leaf_co2_ppress = model_inputs['co2_ppress']
            leaf_co2_ppress = torch.clamp(leaf_co2_ppress, min=1e-6)
            can_co2_ppress  = torch.clamp(leaf_co2_ppress + h2o_co2_bl_diffuse_ratio / model_inputs['gb_mol'] * a_gs * model_inputs['can_press'],min=1.e-06)
        else:
            can_co2_ppress  = model_inputs['co2_ppress']
            leaf_co2_ppress = can_co2_ppress  -  h2o_co2_bl_diffuse_ratio / model_inputs['gb_mol'] * a_gs * model_inputs['can_press']
            leaf_co2_ppress = torch.clamp(leaf_co2_ppress, min=1e-6)


        term_gsmol = h2o_co2_stoma_diffuse_ratio * anet / (leaf_co2_ppress / model_inputs['can_press'])
        #####################################################################
        if self.flag == "global":
            model_inputs['vpd'] = torch.clamp(model_inputs['vpd'], min = 0.001*50)
        #####################################################################
        aquad = 1.0

        bquad = -(2.0 * (model_inputs['stomatal_intercept_btran_Med'] + term_gsmol) + (model_inputs['medlyn_slope'] * term_gsmol) ** 2 / (
                model_inputs['gb_mol'] * model_inputs['vpd']))

        cquad = model_inputs['stomatal_intercept_btran_Med'] * model_inputs['stomatal_intercept_btran_Med']+ \
                (2.0 * model_inputs['stomatal_intercept_btran_Med'] + term_gsmol * (1.0 - model_inputs['medlyn_slope'] * model_inputs['medlyn_slope'] / model_inputs['vpd'])) * term_gsmol

        # gs_mol computation updated here to avoid nan values when gs_mol quadratic function gives complex roots
        gs_mol = model_inputs['stomatal_intercept_btran_Med'].clone()
        mask = (anet < 0.0) | (bquad * bquad < 4.0 * aquad * cquad)
        mask = mask.type(torch.bool)
        gs_mol[torch.logical_not(mask)] = quadratic_max(aquad, bquad[torch.logical_not(mask)],cquad[torch.logical_not(mask)])

        return anet, gs_mol, can_co2_ppress

    def predict_An_gs(self,
                      ftol_epoch,
                      x_in,
                      option        =0):

        # First: Solve the nonlinear system using learned pp1 and pp2 values for x or Ci
        model_inputs = self.pre_fates()
        f            = nonlinearsolver(self)
        J1           = Jacobian(mtd="batchScalarJacobian_AD")

        if ftol_epoch < 1e-3:
            x0 = x_in.detach()
            x0.requires_grad_(True)
        else:
            x0 = self.get_guess(model_inputs)
            x0.requires_grad_(True)

        if option == 0:
            vG   = tensorNewton(f, J1, settings={"maxiter": 10, "ftol": 1e-6, "xtol": 1e-6, "alpha": 0.75})
            x    = vG(x0)
            ftol = f(x)
            if ftol.abs().max() > 1e-4:
                vG   = tensorNewton(f, J1, mtd="rtsafe", lb=0, ub=1000,settings={"maxiter": 70, "ftol": 1e-6, "xtol": 1e-6, "alpha": 0.75})
                x    = vG(x0)
                ftol = f(x)
        else:
            vG   = tensorNewton(f, J1, settings={"maxiter": 10, "ftol": 1e-6, "xtol": 1e-6, "alpha": 0.75})
            x    = vG(x0)
            ftol = f(x)

        # Second: Forward the model using the solution for x == Ci (intercellular leaf CO2 pressure)
        anet, gs_mol,_ = self.forward_model(model_inputs,x)
        model_outputs  = dict()
        model_outputs['Photo'] = anet
        model_outputs['Cond']  = gs_mol / 10 ** 6  # convert to mol m-2 s-1
        return model_outputs, ftol, x


class soilwater_stress_model(torch.nn.Module):
    def __init__(self,args, use_ST=False):
        super(soilwater_stress_model, self).__init__()
        self.device = args['device']
        self.nly    = args['nly']
        self.use_ST = use_ST
    def get_btran_params(self, df):
        # Description: calculate btran : soil water stress factor, where btran should be > 0 and < 1
        # using the B values learned by NN_B

        # Inputs:
        # df    : dataframe with the required datasets

        # Outputs:
        # a dictionary with parameters used to calculate btran including:
            # r      : plant root distribution based on PFT
            # SMC    : soil wettness
            # epsi_o : soil matric potential for open stomata
            # epsi_c : soil matric potential for closed stomata

        r  = np.empty((self.nly  , len(df)));
        SMC= np.empty((self.nly, len(df)));
        if self.use_ST:
            ST = np.empty((self.nly, len(df)))
            SM = np.empty((self.nly, len(df)))
        for ly in range(self.nly):
            SMC[ly] = df[f'SMC_{ly}'].values
            r[ly]   = df[f'r{ly + 1}'].values
            if self.use_ST:
                ST[ly] = df[f'ST_{ly}'].values
                SM[ly] = df[f'SM_{ly}'].values

        r   = torch.FloatTensor(r).to(self.device)
        SMC = torch.FloatTensor(SMC).to(self.device)
        if self.use_ST:
            ST = torch.FloatTensor(ST).to(self.device)
            SM = torch.FloatTensor(SM).to(self.device)

        epsi_o = torch.FloatTensor(df.epsi_o.values).to(self.device)
        epsi_c = torch.FloatTensor(df.epsi_c.values).to(self.device)

        # create dictionary instead of self
        # refined the size of epsi_o and epsi_c to be [nly, dataset size] as r, SMC , ST and SM
        btran_params = dict()
        btran_params['r']       = r
        btran_params['SMC']     = SMC
        if self.use_ST:
            btran_params['ST']      = ST
            btran_params['SM']      = SM
        btran_params['epsi_o']  = epsi_o.repeat(self.nly,1)
        btran_params['epsi_c']  = epsi_c.repeat(self.nly,1)

        return btran_params

    # now takes params as input as well beside B
    def get_btran(self, B, params):
        # Description: calculate btran : soil water stress factor, where btran should be > 0 and < 1
        # using the B values learned by NN_B

        # Inputs:
        # params: parameters used in btran calculations from "get_btran_params" function
        # B     : parameter B values learnt by B neural network

        # Outputs:
        # btran: The soil water stress factor (btran) based on learned B values by NN_B

        params['r'] = [1.0] if self.nly == 1 else params['r']

        # epsi_o and epsi_c now have the sizes of nly X dataset size so .reshape(-1) was added
        epsi = torch.max(params['epsi_o'].reshape(-1)  * params['SMC'].reshape(-1) ** -B, params['epsi_c'].reshape(-1) )
        epsi = epsi.view(self.nly, -1)

        btran = 0.0
        for ly in range(0, self.nly):
            if self.use_ST:
                mask      = (params['ST'][ly] <= t_water_freeze_k_1atm - 273.15 - 2) | (params['SM'][ly] <= 0.0)
                mask      = mask.type(torch.bool)
                # epsi_o and epsi_c were updated to epsi_c[ly] and epsi_o[ly] since now have the sizes of nly X dataset size
                wilt_fact = torch.where(mask, 0.0, torch.clamp((params['epsi_c'][ly] - epsi[ly]) / (params['epsi_c'][ly] - params['epsi_o'][ly]), max=1.0))
            else:
                wilt_fact = torch.clamp((params['epsi_c'][ly] - epsi[ly]) / (params['epsi_c'][ly] - params['epsi_o'][ly]), max=1.0)
            btran += params['r'][ly] * wilt_fact

        btran = torch.clamp(btran, min=0.0)

        return btran
