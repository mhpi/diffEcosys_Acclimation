import copy
import os
import json
import numpy as np
from pathlib import Path
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    print("YAML Module not found.")
    exit(1)  # Exit the program if YAML is not available

def get_pft_dicts(args):
    # Description: A function to prepare lists of plant functional types (dependent on data availability)

    with open(Path(__file__).parent.parent / "conf/PFTs.json") as json_file:
        PFTs_master = json.load(json_file)

    pft_lst      = PFTs_master['pft_lst']
    pft_glob_lst = PFTs_master['pft_glob_lst']

    pft_glob_dict = {k: v for k, v in zip(pft_glob_lst, np.arange(len(pft_glob_lst)))}
    pft_glob_dict_filtered = {k: v for k, v in pft_glob_dict.items() if k in pft_lst}

    args['pft_lst']                = pft_lst
    args['pft_glob_dict']          = pft_glob_dict
    args['pft_glob_dict_filtered'] = pft_glob_dict_filtered

    return args

def get_exp_args(args):
    # Description: A function to prepare some fields in the config file (args) according to the chosen experiment,
    # these fields include:

    # cont_cols_v: which environment factors to be used as input to NN_alpha
    # env_flag   : a flag (true: means use LNC or environmental factors, false: mean use PFT only)
    # out_v      : number of parameters to be learnt by NNv and NNalpha
    # param_list : list of physical parameters to be learnt by NNv
    # param_exp_name: experiment name based on which parameters to be learnt
    # param_range_list: list of min and max bounds for each parameter learnt

    with open(Path(__file__).parent.parent / "conf/master_exp.json") as json_file:
        master = json.load(json_file)

    params_minmax     = master['params_minmax']
    NN_INconfig_dict  = master['NN_INconfig_dict']
    param_exp_dict    = master['param_exp_dict']

    cont_cols, env_flag               = NN_INconfig_dict[args['NNconfig_name']]
    out_v, param_exp_name, param_list = param_exp_dict[args['param_exp']]
    param_range_list = []

    for pr in param_list:
        param_range_list.append(tuple(params_minmax[pr]))

    args['cont_cols_v']     = cont_cols
    args['env_flag']        = env_flag
    args['out_v']           = out_v
    args['param_exp_name']  = param_exp_name
    args['param_list']      = param_list
    args['param_range_list']= param_range_list

    return args

def get_NNv_args(args):
    # Description: A function to specify the features of NNv:
    # Inv is the input size, hd_v: is the hidden size of each hidden layer
    pft_lst = args['pft_lst']

    args['In_v'] = len(pft_lst)
    args['hd_v'] = len(pft_lst)
    # out_sz_v defined in get_exp_args
    return args

def get_NNB_args(args):
    # Description: A function to specify the features of NNB:
    # In_b is the input size, hd_b: is the hidden size of each hidden layer, out_b: is the output size

    args['cont_cols_B'] = [f'soil_clay', f'soil_sand', f'soil_OM']
    args['In_b']        =  3
    args['hd_b']        =  8  # (embeddings _+ input size)
    args['out_b']       =  1
    return args

def get_NNalpha_args(args):
    # Description: A function to specify the features of NNalpha

    args['hd_alpha'] = len(args['pft_lst']) + len(args['cont_cols_v'])
    args['lb_alpha'] =  0.5
    args['ub_alpha'] =  1.5
    args['plty']     =  1.5
    return args

def get_stat_dicts(args):
    with open(Path(__file__).parent.parent / "data/statnorm.json") as json_file:
        stat_dicts = json.load(json_file)
        args['stat_dict'] = stat_dicts['stat_LGE_V']
    return args

def create_output_dir(args):
    # Description: A function to create the output directory

    Action = args['Action']
    action_paths = {
        0: "All",
        1: "Spatial",
        2: "Temporal"
    }

    base_path = args['output_path']
    nn_config = args['NNconfig_name']
    param_exp = args['param_exp']
    run_case  = args['run_case']
    trained_models_path = args['trained_models_path']

    if Action == 3:
        args['output_path'] = os.path.join(base_path, f"{nn_config}/{param_exp}/{run_case}/")
        args['trained_models_path'] = os.path.join(trained_models_path,f"All/{nn_config}/{param_exp}/{run_case}")
    else:
        args['output_path'] = os.path.join(base_path, f"{action_paths[Action]}/{nn_config}/{param_exp}/{run_case}/")

    if args['NNconfig_name'] == "PFT":
        if Action == 3:
            args['trained_models_path'] = os.path.join(args['trained_models_path'], "models")
    else:
        p_value = f"p={args['plty']}"
        args['output_path']         = os.path.join(args['output_path'], p_value)
        if Action == 3:
            args['trained_models_path'] = os.path.join(args['trained_models_path'], p_value, "models")

    os.makedirs(args['output_path'], exist_ok = True)
    return args

def set_filename(args):
    # Description : A function to set the output file name according
    # to chosen parameter experiment and input configuration
    args['fp_name'] = f"{args['param_exp_name']}_{args['NNconfig_name']}_sed="
    return args


def set_config_args(save=True):
    # Description: A function to build and save the run configuration file
    # Initialize YAML parser
    yaml = YAML(typ="safe")

    # Path to the YAML file (update if necessary)
    config_path_NN_model = "conf/config.yaml"

    # Convert relative path to absolute path
    path_NN_model = Path(__file__).parent.parent / config_path_NN_model

    # Check if the file exists before trying to open
    if not os.path.exists(path_NN_model):
        print(f"Config file not found: {path_NN_model}")
        exit(1)

    # Read the YAML file
    with open(path_NN_model, "r") as stream_NN_model:
        args = yaml.load(stream_NN_model)

    # Run the setup functions
    args = get_pft_dicts(args)
    args = get_exp_args(args)
    args = get_NNv_args(args)
    args = get_NNB_args(args)
    args = get_NNalpha_args(args)
    args = get_stat_dicts(args)
    args = create_output_dir(args)
    args = set_filename(args)


    # saving the config file in output directory
    args_to_save = copy.deepcopy(args)
    del args_to_save['pft_glob_dict']
    del args_to_save['pft_glob_dict_filtered']
    if save:
        config_file = json.dumps(args_to_save)
        config_path = os.path.join(args['output_path'], "config_file.json")
        if os.path.exists(config_path):
            os.remove(config_path)
        f = open(config_path, "w")
        f.write(config_file)
        f.close()
    return args

if __name__ == "__main__":
    set_config_args()