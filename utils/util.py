import os
import copy
import glob
import xarray as xr
from data.Stat import *

def clearnan(x):
    # Description: A function to clear nan from x array
    return x[~np.isnan(x)]

def set_seed_device(args,seed=None):
    # Description: A function to set the seed for results reproduction
    if seed == None:
        seed = args['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args['device'])
    return

def get_trained_models_paths(args):
    # Description: A function to get the paths of all pre-trained models including:
    # Vmodels (NNv), Bmodels (NNB), ALmodels (NNalpha)
    def get_paths(model_type):
        return glob.glob(os.path.join(args['trained_models_path'], f"{model_type}models", "*.pt"))
    return get_paths("V"), get_paths("B"), get_paths("AL")

def get_trained_models(args, models):
    # Description: A function to load all pretrained model using the trained model paths

    trained_models_paths = get_trained_models_paths(args)
    trained_models       = []
    for model, paths in zip(models, trained_models_paths):
        models_list = []
        paths       = sorted(paths)
        for path in sorted(paths):
            # Load the saved model and freeze its weights
            trained_model = copy.deepcopy(model)
            trained_model.load_state_dict(torch.load(path))
            for param in trained_model.parameters():
                param.requires_grad = False

            trained_model.eval()
            models_list.append(trained_model)
        trained_models.append(models_list)
    return  trained_models

def create_combinedDS(path, reverse = False, var = "data"):
    # Description: A function to create a combined xarray from two xarrays one for the northern hemisphere and
    # one for the southern hemisphere

    ds_NH    = xr.open_dataset(path)
    ds_SH = xr.open_dataset(path.replace("JJA", "DJF"))
    combined_data = np.zeros_like(ds_SH[f"{var}"].values) * np.nan
    sub_id        = int(combined_data.shape[0]/2)
    if reverse:
        combined_data[:sub_id] = ds_SH[f"{var}"].values[:sub_id];
        combined_data[sub_id:] = ds_NH[f"{var}"].values[sub_id:]
    else:
        combined_data[:sub_id] = ds_NH[f"{var}"].values[:sub_id];
        combined_data[sub_id:] = ds_SH[f"{var}"].values[sub_id:]
    return combined_data

def get_pftmask(args, pft_global_file, pft, per):
    pft_data = xr.open_dataset(pft_global_file)
    pft_mask_x = np.where(pft_data.PCT_PFT_LAND.isel(natpft=args['pft_glob_dict_filtered'][pft]).values[::-1] > per)[0]
    pft_mask_y = np.where(pft_data.PCT_PFT_LAND.isel(natpft=args['pft_glob_dict_filtered'][pft]).values[::-1] > per)[1]
    return pft_mask_x, pft_mask_y

def process_range(scale_params, device):
    # Convert scale_params to tensors for min and range
    min_vals   = torch.tensor([x[0] for x in scale_params])
    range_vals = torch.tensor([x[1] - x[0] for x in scale_params])
    max_vals   = torch.tensor([x[-1] for x in scale_params])

    min_vals   = min_vals.unsqueeze(0)    # Shape [1, nparams]
    range_vals = range_vals.unsqueeze(0)  # Shape [1, nparams]
    max_vals   = max_vals.unsqueeze(0)    # Shape [1, nparams]

    return min_vals.to(device), range_vals.to(device), max_vals.to(device)













