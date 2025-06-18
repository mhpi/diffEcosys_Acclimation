import os
import glob
import torch
import numpy as np
import xarray as xr
import pandas as pd
from data.Stat import cal_stats
class output_class():
    # Description: A class for post processing after training and testing the differentiable model
    def __init__(self, args):
        self.var_list    = args['var_cols']
        self.par_list    = args['param_list']
        self.target_list = args['target_cols']

        self.varpar_list     = []
        for item in self.var_list + self.par_list:
            if item in self.par_list:
                self.varpar_list.append(f'{item}_learnt')
            elif item in self.var_list:
                self.varpar_list.append(f'{item}_sim')
            else:
                raise ValueError(f"{item} is not in variables or parameters lists")
    def get_results(self, df, results):
        for item_name in self.varpar_list:
            # Try to get the value from var_results, if not found, get it from param_results
            value = results.get(item_name.split('_')[0], None)
            df = df.reset_index(drop = True)
            df[f'{item_name}'] = value
        return df

    def get_metrics(self, obs_all, suffix, *sim_all):
        metrics = pd.DataFrame()

        # Iterate over each item in the target list
        for item in self.target_list:
            try:
                obs = obs_all[item].cpu().numpy()
            except:
                obs = obs_all[item]
            for sim_dict in sim_all:
                try:
                    sim = sim_dict[item]
                    mask = np.where(np.logical_and(~np.isnan(sim), ~np.isnan(obs)))[0]
                    metrics[f'{suffix}_{item}'] = cal_stats(sim[mask], obs[mask])
                except:
                    continue

        return metrics
    def save_results(self, args, df):
        output_path = os.path.join(args['output_path'], "Results")
        if not (os.path.exists(output_path)):
            os.makedirs(output_path)
        df.to_csv(os.path.join(output_path, f"test_{args['fp_name']}.csv"), index=False)
        return
    def save_metrics(self, args, metrics):
        output_path = os.path.join(args['output_path'], "metrics")
        if not (os.path.exists(output_path)):
            os.makedirs(output_path)
        metrics.to_csv(os.path.join(output_path, f"metrics_{args['fp_name']}.csv"))
    def save_models(self, args, trained_models):
        V_model, B_model, alpha_model = trained_models
        output_path = os.path.join(args['output_path'], "models")
        if not (os.path.exists(output_path)):
            os.makedirs(os.path.join(output_path, "Vmodels"))
            os.makedirs(os.path.join(output_path, "Bmodels"))
            os.makedirs(os.path.join(output_path, "ALmodels"))

        # it is a list
        torch.save(V_model[0].state_dict(), os.path.join(output_path, "Vmodels/Vmod_{}.pt".format(args['fp_name'])))
        torch.save(B_model[0].state_dict(), os.path.join(output_path, "Bmodels/Bmod_{}.pt".format(args['fp_name'])))
        torch.save(alpha_model[0].state_dict(), os.path.join(output_path, "ALmodels/ALmod_{}.pt".format(args['fp_name']))) if len(alpha_model)>0 else None

        return

    def combine_results_from_dicts(self, *dicts):
        combined_dict = {}
        # Iterate over all keys in the first dictionary
        for key in dicts[0]:
            combined_values = []

            # Iterate through all dictionaries to get values for the same key
            for d in dicts:
                if key in d:
                    # combined_values += list(d[key])  # Combine as lists
                    combined_values.append(d[key])

            # Convert to NumPy array if the original data is NumPy arrays
            if isinstance(dicts[0][key], np.ndarray):
                combined_dict[key] = np.concatenate(combined_values)
            elif isinstance(dicts[0][key], torch.Tensor):
                combined_dict[key] = torch.concatenate(combined_values)
            else:
                combined_dict[key] = combined_values

        return combined_dict
    def chunk_metrics_folds(self, metrics_folds: object, suffix: object) -> object:
        metrics_all = pd.DataFrame()
        for var in self.target_list:
            metrics_all[f'{suffix}_{var}'] = metrics_folds.filter(like =f'{suffix}_{var}').mean(axis = 'columns').values
            metrics_all.index = ['COR', 'RMS', 'Bias', 'NSE', 'KGE']
        return metrics_all


    def concise_results(self, df, filter_columns):
        return df[filter_columns + self.varpar_list]

    def df_to_xarray(self, args, df):
        self.xarray_list = []
        for col in self.varpar_list:
            if col in df.columns:
                out = np.zeros((args['lat_range'], args['lon_range'])) * np.nan
                valid_rows = df['latitude_idx'].values
                valid_cols = df['longitude_idx'].values
                out[valid_rows, valid_cols] = df[col].values
                out_xarray = xr.DataArray(data=out,dims=["y", "x"],
                    coords=dict(
                        x=(["x"], np.linspace(args['lon_start'], args['lon_end'], args['lon_range'])),
                        y=(["y"], np.linspace(args['lat_start'], args['lat_end'], args['lat_range'])),
                        band=1,
                    ),
                    attrs=dict(
                        AREA_OR_POINT='Area',
                        scale_factor=1.0,
                        add_offset=0.0,

                    ))
                self.xarray_list.append(out_xarray)
            else:
                raise KeyError(f"{col} not found in your dataframe")


        return # self.xarray_list

    def resample_save_xarray(self, args, pft, month, resample = True):
        target_array = xr.open_dataset(args['pft_global_path'])

        for item, ds in zip(self.varpar_list,self.xarray_list):
            if resample:
                # Define the target resolution (lower resolution)
                new_resolution = {
                    'x': np.linspace(ds.x.min().item(), ds.x.max().item(), len(target_array.lsmlon.values)),
                    'y': np.linspace(ds.y.max().item(), ds.y.min().item(),len(target_array.lsmlat.values))}
                # Specify the new latitude and longitude values

                # Resample the dataset to the target resolution
                array_resampled          = ds.interp(coords=new_resolution, method='linear')
                array_resampled_ds       = array_resampled.to_dataset(name = 'data')#{item}
                array_resampled_ds.attrs = array_resampled.attrs
                array_resampled_ds.to_netcdf(os.path.join(args['output_path'], f"per_PFT/{item}_{month}_{pft}.nc"))
            else:
                ds = ds.to_dataset(name = 'data')#{item}
                ds.attrs = ds.attrs
                ds.to_netcdf(os.path.join(args['output_path'], f"per_PFT/{item}_{month}_{pft}.nc"))

        return

    def combine_pfts_togrid(self, args, month, ext = ""):
        # read the pft data for upscaling
        pft_data = xr.open_dataset(args['pft_global_path'])

        for item in self.varpar_list:
            # some initializations
            pft_global_percentage_all = 0.0
            pft_output_combined = 0.0
            pft_output_dict = {}
            # pft_output_dict_higres = {}
            for pft in args['pft_lst']:
                pft_output = xr.open_dataset(os.path.join(args['output_path'], f"per_PFT/{item}_{month}_{pft}.nc"))
                pft_output_dict[f'{pft}'] = (['y', 'x'], pft_output.data.values)
                os.remove(os.path.join(args['output_path'], f"per_PFT/{item}_{month}_{pft}.nc"))
                # multiply each PFT global prediction by its global percentage
                pft_global_percentage      = pft_data.PCT_PFT_LAND.isel(natpft= args['pft_glob_dict_filtered'][pft]).values[::-1]

                pft_output_combined       += (pft_output.data.values * pft_global_percentage)#{item}
                pft_global_percentage_all += pft_data.PCT_PFT_LAND.isel(natpft=args['pft_glob_dict_filtered'][pft]).values[::-1]


            if 'sim' in item:
                pft_output_combined_data = pft_output_combined / 100#pft_global_percentage_all#
            else:
                pft_output_combined_data = pft_output_combined / pft_global_percentage_all
            pft_output_ds = xr.Dataset(pft_output_dict,  coords = { 'y': pft_output.y.values, 'x': pft_output.x.values,})
            pft_output_combined_ds = xr.DataArray(data= pft_output_combined_data, dims=["y", "x"],
                                     coords=dict(
                                     x=(["x"], pft_output.x.values),
                                     y=(["y"], pft_output.y.values),
                                     band=1,),attrs=dict(
                                     AREA_OR_POINT='Area',
                                     scale_factor=1.0,
                                     add_offset=0.0)).to_dataset(name = 'data')#{item}

            pft_output_ds.to_netcdf(os.path.join(args['output_path'], f"per_PFT/{item}_{month}_per_PFT{ext}.nc"))
            pft_output_combined_ds.to_netcdf(os.path.join(args['output_path'], f"gridded/{item}_{month}_combined{ext}.nc"))


    def chunk_griddedfolds(self, output_path, month):

        for item in self.varpar_list:
            files    = glob.glob(os.path.join(output_path, f"gridded/{item}_{month}_combined_fold*.nc"))
            datasets = [xr.open_dataset(file) for file in files]

            # Concatenate datasets along a new dimension
            combined_ds = xr.concat(datasets, dim='folds')

            # Compute the mean and standard deviation across the new dimension
            mean_ds = combined_ds.mean(dim='folds')
            std_ds  = combined_ds.std(dim='folds')

            mean_ds.to_netcdf(os.path.join(output_path, f"gridded/{item}_{month}_combined_mean.nc"))
            std_ds.to_netcdf(os.path.join(output_path, f"gridded/{item}_{month}_combined_std.nc"))

            for file in files:
                os.remove(file)
