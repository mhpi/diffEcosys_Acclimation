import os
import torch
import pandas as pd
from data.load_data                  import load_data
from models.Train_Test               import Train_diffmodel, Test_diffmodel
from models.physical.physical_models import leaflayerphotosynthesis_model, soilwater_stress_model
from utils.NNmodels_utils            import get_nnmodels, get_nnparams
from utils.util                      import set_seed_device, get_trained_models
from utils.post_utils                import output_class

def main_All(args):
    args['epochs'] = args['epochs_all']
    fp_name        = args['fp_name']

    for n, seed in enumerate([42] + args['seed_lst']):
        set_seed_device(args, seed)
        args['fp_name'] = fp_name + f"{n-1}"

        # ==========Initialize========
        output_instance  = output_class(args)

        # ============Data==========
        dataloader   = load_data(args)
        LGE_data     = dataloader.LGE_data
        vcmax25_data = dataloader.vcmax25_data
        data_dict    = dataloader.create_data_dict(args, LGE_data, vcmax25_data, mtd=0)

        # ===========Models==========
        models_dict = dict()
            # physical component
        photo_model  = leaflayerphotosynthesis_model()
        soil_model   = soilwater_stress_model(args)
        models_dict['physical_models'] = [photo_model, soil_model]

            # NN component
        models_dict['nn_models'] = get_nnmodels(args)
        nn_params                = get_nnparams(models_dict['nn_models'])
        optim                    = torch.optim.Adam(nn_params, lr=args['lr'])

        # ============TrainTest==========
        models_dict = Train_diffmodel(args, data_dict, models_dict, optim)
        sim_LGE, sim_other = Test_diffmodel(args, data_dict, models_dict)

        if n>0:
            results  = output_instance.get_results(data_dict['LGE_data'], sim_LGE)
            metrics = output_instance.get_metrics(data_dict['target_data'] , "All",sim_LGE, sim_other)
            print(metrics)
            output_instance.save_results(args, results)
            output_instance.save_metrics(args, metrics)
            output_instance.save_models(args, models_dict['trained_nn_models'])

    return

def main_spatial(args):
    print(args['param_exp'])
    args['epochs'] = args['epochs_spatial']
    fp_name        = args['fp_name']
    for n, seed in enumerate([42] + args['seed_lst']):
        args['fp_name'] = fp_name + f"{n-1}"

        # ============Data==========
        dataloader = load_data(args)
        LGE_data        = dataloader.LGE_data
        vcmax25_data    = dataloader.vcmax25_data

        locs_list    = args['locs_list']

        # ==========Initialize========
        output_instance  = output_class(args)
        metrics_train_mean  = pd.DataFrame()
        sim_test_all =  {}
        obs_test_all =  {}
        df_test_all  = pd.DataFrame()
        for loc in locs_list:

            # ============Data==========
            test, train = dataloader.extract_by_locs(LGE_data, [loc])
            data_dict_train = dataloader.create_data_dict(args, train, vcmax25_data, mtd = 0)
            data_dict_test  = dataloader.create_data_dict(args, test, vcmax25_data, mtd = 1)

            # ===========Models==========
            models_dict     = dict()
            # physical component
            photo_model = leaflayerphotosynthesis_model()
            soil_model = soilwater_stress_model(args)
            models_dict['physical_models'] = [photo_model, soil_model]

            # NN component
            set_seed_device(args, seed)
            models_dict['nn_models'] = get_nnmodels(args)
            nn_params = get_nnparams(models_dict['nn_models'])
            optim = torch.optim.Adam(nn_params, lr=args['lr'])

            # ============TrainTest==========
            models_dict = Train_diffmodel(args, data_dict_train, models_dict, optim)
            sim_LGE_train, sim_other_train = Test_diffmodel(args, data_dict_train, models_dict)
            sim_LGE_test , sim_other_test  = Test_diffmodel(args, data_dict_test, models_dict)

            # ============RESULTS==========
            # TEST
            test        = output_instance.get_results(test, sim_LGE_test)
            df_test_all = pd.concat([df_test_all, test])

            if len(sim_test_all) == 0:
                sim_test_all = sim_LGE_test.copy()
            else:
                sim_test_all = output_instance.combine_results_from_dicts(sim_test_all,sim_LGE_test )


            if len(obs_test_all) == 0:
                obs_test_all = data_dict_test['target_data'].copy()
            else:
                obs_test_all = output_instance.combine_results_from_dicts(obs_test_all,data_dict_test['target_data'] )

            # TRAIN
            metrics_train      = output_instance.get_metrics(data_dict_train['target_data'], "train", sim_LGE_train, sim_other_train)
            metrics_train_mean = pd.concat([metrics_train_mean, metrics_train],axis=1)
            print('For Test loc = ', loc)
            print(" TRAINING METRICS")
            print(metrics_train)
            print(" TESTING METRICS")
            print(output_instance.get_metrics(obs_test_all, "test", sim_test_all))

        # ============SAVE==========
        if n> 0:
            Results_metrics = output_instance.chunk_metrics_folds(metrics_folds = metrics_train_mean, suffix='train')
            metrics_test    = output_instance.get_metrics(obs_test_all, "test", sim_test_all)
            Results_metrics = pd.concat([Results_metrics,metrics_test], axis = 1)
            output_instance.save_metrics(args, Results_metrics)
            output_instance.save_results(args, df_test_all)

    return

def main_temporal(args):
    args['epochs'] = args['epochs_all']
    fp_name        = args['fp_name']
    # ============Data==========
    dataloader   = load_data(args)
    LGE_data     = dataloader.LGE_data
    vcmax25_data = dataloader.vcmax25_data

    # ==========Initialize========
    output_instance = output_class(args)
    Results_metrics = pd.DataFrame()

    for n, seed in enumerate([42] + args['seed_lst']):
        args['fp_name'] = fp_name + f"{n-1}"

        # ============Data==========
        train, test     = dataloader.extract_by_time(LGE_data, args['time_ratio'])
        data_dict_train = dataloader.create_data_dict(args, train, vcmax25_data, mtd=0)
        data_dict_test  = dataloader.create_data_dict(args, test, vcmax25_data, mtd=1)

        # ===========Models==========
        models_dict = dict()
        # physical component
        photo_model = leaflayerphotosynthesis_model()
        soil_model = soilwater_stress_model(args)
        models_dict['physical_models'] = [photo_model, soil_model]

        # NN component
        set_seed_device(args, seed)
        models_dict['nn_models'] = get_nnmodels(args)
        nn_params = get_nnparams(models_dict['nn_models'])
        optim = torch.optim.Adam(nn_params, lr=args['lr'])

        # ============TrainTest==========
        models_dict = Train_diffmodel(args, data_dict_train, models_dict, optim)
        sim_LGE_train, sim_other_train = Test_diffmodel(args, data_dict_train, models_dict)
        sim_LGE_test, sim_other_test = Test_diffmodel(args, data_dict_test, models_dict)
        if n>0:
            metrics_train = output_instance.get_metrics(data_dict_train['target_data'], "train", sim_LGE_train)
            metrics_test  = output_instance.get_metrics(data_dict_test['target_data'], "test", sim_LGE_test)
            output_instance.save_metrics(args, pd.concat([metrics_train, metrics_test], axis=1))
            Results_metrics = pd.concat([Results_metrics, metrics_train, metrics_test], axis=1)

    args['fp_name'] = fp_name + "All"
    Results_metrics = Results_metrics.groupby(Results_metrics.columns, axis=1).mean()
    output_instance.save_metrics(args, Results_metrics)
    return

def main_global(args):
    years       = args['global_forward']['yr_Range']
    seasons     = args['global_forward']['seasons']
    outPath     = args['output_path']
    input_path  = args['Global_data_path']
    models_dict = dict()

    # physical component
    photo_model = leaflayerphotosynthesis_model(flag = args['spatial_flag'])
    soil_model  = soilwater_stress_model(args)
    models_dict['physical_models'] = [photo_model, soil_model]

    # NN component
    models_dict['trained_nn_models'] = get_trained_models(args, get_nnmodels(args))

    for year in range(years[0], years[1]+1):
        for seaSon in seasons:

            # Create required child directories
            args['output_path'] = os.path.join(outPath,f"{year}/{seaSon}/")
            if not (os.path.exists(args['output_path'])):
                os.makedirs(os.path.join(args['output_path'], "per_PFT"))
                os.makedirs(os.path.join(args['output_path'], "gridded"))

            # Initialize output instance
            output_instance = output_class(args)

            # loop through pfts
            for pft in args['pft_lst']:
                print(pft)
                df_results   = pd.DataFrame()

                args['Global_data_path'] = os.path.join(input_path, f"{year}/{seaSon}/df_glob_{seaSon}_{pft}.feather")
                dataloader   = load_data(args )
                df_glob_PFT  = dataloader.glob_data
                df_batches   = dataloader.create_batches(df_glob_PFT, batch_size= 20000)
                for m in range(len(df_batches)):
                    df_batches[m].reset_index(drop=True, inplace=True)
                    data_dict = dataloader.create_data_dict(args, df_batches[m], None, mtd=2)

                    sim_glob, _ = Test_diffmodel(args, data_dict, models_dict)

                    df_batches[m] = output_instance.get_results(data_dict['LGE_data'],sim_glob)
                    df_batches[m] = output_instance.concise_results(df_batches[m],['latitude_idx','longitude_idx','latitude','longitude'])
                    # ==================================================================================================
                    df_batches[m] = df_batches[m].reset_index(drop = True); ftol    = sim_glob['ftol']
                    df_batches[m]['ftol'] = ftol.detach().to('cpu').numpy()
                    df_results         = pd.concat([df_results, df_batches[m]])
                # ======================================================================================================
                output_instance.df_to_xarray(args, df_results)
                output_instance.resample_save_xarray(args, pft, seaSon)
                print(f'Done {pft}')

            output_instance.combine_pfts_togrid(args, seaSon)
    return





