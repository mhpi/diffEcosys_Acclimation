import pandas as pd
from data.Stat       import scale
from data.data_split import create_sets_temporal
from models.physical.physical_models import *

class load_data():

    # A data loading class that reads input data based on the input configuration.
    #
    # Parameters
    # ----------
    # args : dict
    #     A dictionary of configuration arguments. Expected keys:
    #         - 'pft_lst'         : List of plant functional types (PFTs) in the data.
    #         - 'device'          : Device for tensor operations (e.g., 'cpu' or 'cuda').
    #         - 'spatial_flag'    : Either 'global' or site specific.
    #         - 'Global_data_path': Path to the global dataset.
    #         - 'LGE_path'        : Path to the lead gas exchange dataset (LGE).
    #         - 'vcmax25_path'    : Path to the Vcmax25 dataset .
    #
    # to_frac : bool, optional (default=True)
    #     If True, indicates that (soil sand and clay percentages, and relative humidity) input data should be converted to fractional format.

    def __init__(self, args, to_frac=True):
        self.pft_lst = args['pft_lst']
        self.to_frac = to_frac
        self.device  = args['device']

        if args['spatial_flag'] == "global":
            self.glob_data     = self.readfile(args['Global_data_path'])
        else:
            self.LGE_data      = self.readfile(args['LGE_path'])
            self.vcmax25_data  = self.readfile(args['vcmax25_path'])
        return

    def readfile(self,file):
        # Description: A function to read the csv file including the  dataset

        # Inputs:
        # file   : csv file or feather file to be read

        # Output:
        # df    : The dataset dataframe

        # read and keep data points corresponding to PFT_lst
        pft_lst = list(np.sort(self.pft_lst))
        #!==============================
        # accepts either csv or feather file
        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_feather(file)
        #!==============================
        df = df[df['PFT'].isin(pft_lst)]
        df = df.reset_index(drop='True')

        # approximate all Vcmax_CLM45 to two decimal places
        if 'vcmax_CLM45' in df.columns:
            df['vcmax_CLM45'] = round(df['vcmax_CLM45'], 2)

        # convert all percentage data to fraction
        if self.to_frac:
            colnames = df.columns.tolist()
            subs = ['soil_sand', 'soil_clay', 'RH']
            subcols = []
            for sub in subs:
                subcols = subcols + [i for i in colnames if sub in i]
            df[subcols] = df[subcols] / 100

        # apply one hot encoding (see function onehot_PFT)
        df = self.onehot_PFT(df)
        return df

    def onehot_PFT(self, df):
        # Description: A function to one hot encode the Plant functional type (PFT) column in the dataset
        # from categorical values to quantitative values

        # Inputs :
        # df     : The dataset dataframe
        # pft_lst: list of the plant functional type categories to be considered for a specific model run

        df['PFT'] = df['PFT'].astype('category')
        df['PFT'] = df['PFT'].cat.set_categories(self.pft_lst)
        df['PFT'] = df['PFT'].cat.reorder_categories(self.pft_lst)
        return df

    def create_batches(self, df, batch_size):
        # Description: A function to split a large dataframe into sub dataframes (batches of dataframes)
        batches = [df[i:i+batch_size] for i in range(0, len(df), batch_size)]
        return batches

    def extract_by_pfts(self, df, lst):
        df = df[df['PFT'].isin(lst)]
        df = df.reset_index(drop = True)
        return df

    def extract_by_locs(self, df , lst):
        df1 = df[df['Location'].isin(lst)]; df2 = df.drop(df1.index)
        df1 = df1.reset_index(drop = True); df2 = df2.reset_index(drop = True)
        return df1, df2

    def extract_by_time(self, df , ratio):
        df1 , df2 = create_sets_temporal(df, ratio)
        return df1, df2

    def get_forcing_attrs(self,df):
        # Description: This function extracts all the data variables required to run the whole model
        # Inputs     : The whole dataset

        # Outputs    :
        # can_press               :Air pressure NEAR the surface of the leaf (Pa)
        # can_o2_ppress           :Partial pressure of O2 NEAR the leaf surface (Pa)
        # can_co2_ppress          :Partial pressure of CO2 NEAR the leaf surface (Pa)
        # veg_tempk               :Leaf temperature     [K]
        # air_tempk               :Air temperature      [K]
        # RH                      :Relative Humidity fraction
        # parsun_lsl              :Photosynthetic active radiation (umol m-2 s-1) ..( to be mutliplied by fraction absorbed and converted to wm-2 using a factor of 4.6)
        # veg_esat                :saturation vapor pressure at veg_tempk (Pa)
        # air_vpress              :air vapor pressure (Pa)
        # BLCond                  :Boundary layer conductance (mol m-2 s-1)
        # rb                      :Boundary layer resistance (s m-1)
        # c3c4_path_index         :Index for which photosynthetic pathway is active.  C4 = 0,  C3 = 1
        # LAI                     :Leaf Area Index
        # vcmax25_clm             :vcmax25 values PFT based from the community land model 4.5 (umol m-2 s-1)
        # btran_clm               :btran values computed based on physical equations in CLM4.0
        # vpd                     :vapor pressure deficit (kPa)

        # mm_kco2                 :Michaelis-Menten constant for CO2 (Pa)
        # mm_ko2                  :Michaelis-Menten constant for O2 (Pa)
        # co2_cpoint              :CO2 compensation point (Pa)
        # cf                      :conversion factor between molar form and velocity form of conductance and resistance: [umol/m3]
        # gb_mol                  :leaf boundary layer conductance (umol H2O/m**2/s)
        # ceair                   :vapor pressure of air, constrained (Pa)
        # qabs                    :PAR absorbed by PS II (umol photons/m**2/s)

        # stomatal_intercept_btran:water-stressed minimum stomatal conductance (umol H2O/m**2/s)
        # medlyn_slope            :Slope for Medlyn stomatal conductance model method, the unit is KPa^0.5
        model_inputs    = dict()
        can_press       = torch.FloatTensor(df.Patm.values * 1000)
        can_o2_ppress   = 0.209 * can_press
        co2_ppress      = torch.FloatTensor(df.CO2S.values * df.Patm.values / 1000)
        veg_tempk       = torch.FloatTensor(df.Tleaf.values + 273.15)
        air_tempk       = torch.FloatTensor(df.Tair.values + 273.15)
        RH              = torch.FloatTensor(df.RH.values)
        parsun_lsl      = torch.FloatTensor(df.PARin.values / 4.6) * FPAR
        veg_esat        = torch.FloatTensor(QSat(veg_tempk, RH)[0])
        air_vpress      = torch.FloatTensor(QSat(veg_tempk, RH)[1])
        BLCond          = torch.FloatTensor(df.BLCond.values)
        rb              = ((can_press * umol_per_kmol) / (rgas_J_K_kmol * air_tempk)) / (BLCond * 10 ** 6)
        c3c4_pathway    = df.Pathway.values
        c3c4_path_index = torch.IntTensor([1 if c3c4_pathway[i] == "C3" else 0 for i in range(len(c3c4_pathway))])
        LAI             = torch.FloatTensor(df.LAI.values)
        vcmax25_clm     = torch.FloatTensor(df.vcmax_CLM45.values)
        btran_clm       = torch.FloatTensor(df.btran_calculated.values)

        # GetCanopyGasParameters
        mm_kco2, mm_ko2, co2_cpoint, cf, gb_mol, ceair = GetCanopyGasParameters(can_press, can_o2_ppress,
                                                                                veg_tempk, air_tempk,
                                                                                air_vpress, veg_esat, rb)

        vpd = torch.FloatTensor(df.VPD.values)

        # compute qabs
        qabs = parsun_lsl * 0.5 * (1.0 - fnps) * 4.6

        # Get Stomatal Conductance parameters
        stomatal_intercept_btran = torch.FloatTensor([stomatal_intercept[i] for i in c3c4_path_index])
        medlyn_slope             = torch.FloatTensor(df.medslope.values)

        model_inputs['can_press']        = can_press.to(self.device) if self.device != None else can_press
        model_inputs['can_o2_ppress']    = can_o2_ppress.to(self.device) if self.device != None else can_o2_ppress
        model_inputs['co2_ppress']       = co2_ppress.to(self.device) if self.device != None else co2_ppress
        model_inputs['veg_tempk']        = veg_tempk.to(self.device) if self.device != None else veg_tempk
        model_inputs['parsun_lsl']       = parsun_lsl.to(self.device) if self.device != None else parsun_lsl
        model_inputs['veg_esat']         = veg_esat.to(self.device) if self.device != None else veg_esat
        model_inputs['mm_kco2']          = mm_kco2.to(self.device) if self.device != None else mm_kco2
        model_inputs['mm_ko2']           = mm_ko2.to(self.device) if self.device != None else mm_ko2
        model_inputs['co2_cpoint']       = co2_cpoint.to(self.device) if self.device != None else co2_cpoint
        model_inputs['cf']               = cf.to(self.device) if self.device != None else cf
        model_inputs['gb_mol']           = gb_mol.to(self.device) if self.device != None else gb_mol
        model_inputs['ceair']            = ceair.to(self.device) if self.device != None else ceair
        model_inputs['qabs']             = qabs.to(self.device) if self.device != None else qabs
        model_inputs['vpd']              = vpd.to(self.device) if self.device != None else vpd

        model_inputs['c3c4_path_index']  = c3c4_path_index.to(self.device) if self.device != None else c3c4_path_index
        model_inputs['stomatal_intercept_btran'] = stomatal_intercept_btran.to(self.device) if self.device != None else stomatal_intercept_btran
        model_inputs['medlyn_slope']     = medlyn_slope.to(self.device) if self.device != None else medlyn_slope
        model_inputs['LAI']              = LAI.to(self.device) if self.device != None else LAI
        model_inputs['Vcmax25_CLM45']    = vcmax25_clm.to(self.device) if self.device != None else vcmax25_clm
        model_inputs['btran_CLM45']      = btran_clm.to(self.device) if self.device != None else btran_clm
        return model_inputs

    def get_target_data(self, args, LGE_df, vcmax25_df):
        # Description: A function to get target variables from the dataset as defined in the target cols argumnet
        # in the config file

        # Inputs:
        # args       : arguments from config file
        # LGE_df     : Leaf Gas Exchange dataframe
        # vcmax25_df : vcmax25 dataframe


        # Outputs:
        # targets: a dictionary with keys (target_cols), and values (tensors corresponding to each target col)

        targets = dict()
        tarLsT = args['target_cols']
        if len(tarLsT) == 0:
            pass
        else:
            for var in tarLsT:
                if var == "Vcmax25":
                    out_var = torch.tensor(vcmax25_df['Vcmax25'].values, dtype=torch.float, device=self.device)
                    targets['Vcmax25'] = out_var
                else:
                    out_var = torch.tensor(LGE_df[var].values, dtype=torch.float, device=self.device)
                    targets[var] = out_var
        return targets

    def get_NNvalpha_data(self,args,mtd, *dfs, function = None):
        # Description: A function to extract inputs to NNv and NNalpha neural networks

        # Inputs:
        # args       : arguments from config file
        # mtd        : scaling method (see  function)
        # *dfs       : input dataframes or datasets

        # Outputs:
        # cats_encoded: encoded categorical inputs
        # conts_scaled: scaled continuous inputs
        cat_cols  = args['cat_cols']
        cont_cols = args['cont_cols_v']
        if len(cont_cols) > 0:
            # Extract continuous values from all dfs
            conts_list   = [df[cont_cols].values for df in dfs]
            conts        = np.concatenate(conts_list)
            conts_scaled = scale(args, conts, mtd, function) if conts is not None else None
            conts_scaled = torch.tensor(conts_scaled, dtype=torch.float, device=self.device)
        else:
            conts_scaled = None

        cats_list    = [df[cat_cols] for df in dfs]
        cats         = pd.concat(cats_list)
        cats_encoded =  torch.tensor(pd.get_dummies(cats).values, device=self.device).float()
        return cats_encoded, conts_scaled

    def get_NNB_data(self, args, df):
        # Description:prepare the categorical and the continuous inputs for B Neural Network
        # The function assumes that the continuous inputs always include the three soil attributes: %sand, %clay and Fom

        # Inputs:
        # cat_cols: Categorical columns used for B Neural Network
        # df      : dataframe or dataset with the required datasets

        # Outputs:
        # Cats: categorical inputs
        # Conts: continous inputs

        cat_cols    = args['cat_cols']
        cont_cols_B = args['cont_cols_B']
        nly         = args['nly']
        if len(cat_cols) == 0:
            cats = []
        elif isinstance(cat_cols, list):
            cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)
        else:
            cats = df[cat_cols].cat.codes.values
        cats = torch.tensor(cats, dtype=torch.int32, device=self.device)
        # modified to have the size of [nly X datasetsize X 1] (meaningful only for batching)
        cats = cats.repeat(nly, 1, 1)

        # modified to have the size of [nly X datasetsize X len(cont_cols2)] (meaningful only for batching)
        conts = np.empty((nly, df.shape[0], len(cont_cols_B)))

        for ly in range(nly):
            cont_cols_lyr = [f"{col}_{ly}" for col in cont_cols_B]
            conts_lyr     = np.stack([df[col].values for col in cont_cols_lyr], axis=1)
            conts[ly,:,:] = conts_lyr

        conts = torch.tensor(conts, dtype=torch.float, device=self.device)
        return cats, conts

    def create_data_dict(self, args, LGE_df, vcmax25_df, mtd ):
        # Description: a wrapper function to create a dictionary of all input data required for running the
        # differentiable framework i.e. (physical model + neural network)

        # Inputs:
        # LGE_df     : Leaf Gas Exchange dataframe
        # vcmax25_df : vcmax25 dataframe

        # Outputs:
        # data_dict : a dictionary with the following keys:
                    # LGE_data   : leaf gas exchange data
                    # physM_data : forcing and attributes required for the physical model (FATES photosynthesis module)
                    # target_data: data with target variables
                    # NNB_data   : Inputs to neural network (NNB)
                    # NNv_data   : Inputs to neural networks (NNV and NNalpha)

        data_dict = dict()
        data_dict['LGE_data']    = LGE_df
        data_dict['physM_data']  = self.get_forcing_attrs(LGE_df)
        data_dict['NNB_data']    = self.get_NNB_data(args, LGE_df)
        if args['spatial_flag'] != "global":
            data_dict['target_data'] = self.get_target_data(args, LGE_df, vcmax25_df)
            data_dict['NNv_data']    = self.get_NNvalpha_data(args, mtd, LGE_df, vcmax25_df, function="norm")
        else:
            data_dict['NNv_data'] = self.get_NNvalpha_data(args, mtd, LGE_df, function="norm")

        return data_dict



