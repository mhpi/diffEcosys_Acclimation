from models.NN.NN_models import embszs_define
import importlib
from pathlib import Path
def choose_model(model_name):
    spec = importlib.util.spec_from_file_location(model_name, Path(__file__).parent.parent /"models/NN/NN_models.py")
    module = spec.loader.load_module()
    model = getattr(module, model_name)
    return model

def build_model(cfgs):
    # Description: A customized function to build nn model based on type inputs and other nn features

    cat_sz   = cfgs['cat_sz']
    cont_sz  = cfgs['cont_sz']
    out_sz   = cfgs['out_sz']
    hd_size  = cfgs['hd_size']
    dp_ratio = cfgs['dp_ratio']
    model    = choose_model(cfgs['model'])
    device   = cfgs['device']

    if cfgs['mtd'] == 0:   # Categorical input only
        nn_model = model(n_cont=cat_sz, out_sz=out_sz, layers=[hd_size, hd_size], p=dp_ratio)
        nn_model = nn_model.to(device=device)
    elif cfgs['mtd'] == 1: # Continuous input only
        nn_model = model(n_cont=cont_sz, out_sz=out_sz, layers=[hd_size, hd_size], p=dp_ratio)
        nn_model = nn_model.to(device=device)
    elif cfgs['mtd'] == 2: # BOTH
        nn_model = model(n_cont=cat_sz+cont_sz, out_sz=out_sz, layers=[hd_size, hd_size], p=dp_ratio)
        nn_model = nn_model.to(device=device)
    elif cfgs['mtd'] == 3:  # Tabular
        emb_szs = embszs_define(cat_szs  = [cat_sz])
        nn_model = model(emb_szs, n_cont = cont_sz, out_sz=out_sz, layers=[hd_size, hd_size], p=dp_ratio);
        nn_model = nn_model.to(device=device)
    elif cfgs['mtd'] == 4:  # None
        nn_model = None
    else:
        raise ValueError("please select valid method for NN model initialization")
    return nn_model

def get_nnmodels(args):
    # Description: A customized function to build the three nn models representing the nn component in the
    # differentiable framework including NNv, NNB, and NNalpha

    V_model = build_model(cfgs={'cat_sz': args['In_v'] , 'cont_sz': None, 'device': args['device'],
                                'out_sz': args['out_v'], 'hd_size': args['hd_v'], 'dp_ratio': 0.0,
                                'model' : args['Vmodel_name'],
                                'mtd'   : 0})

    B_model = build_model(cfgs={'cat_sz': len(args['pft_lst']), 'cont_sz': args['In_b'], 'device': args['device'],
                                'out_sz': args['out_b']       , 'hd_size': args['hd_b'], 'dp_ratio': args['dp'],
                                'model' : args['Bmodel_name'],
                                'mtd'   : 3})

    alpha_model = build_model(cfgs={'cat_sz'  : len(args['pft_lst']), 'cont_sz': len(args['cont_cols_v']), 'device': args['device'],
                                    'out_sz'  : args['out_v']       ,
                                    'hd_size' : args['hd_alpha'],
                                    'dp_ratio': args['dp'],
                                    'model'   : args['alphamodel_name'],
                                    'mtd'     : 2 if args['env_flag'] else 4})

    return V_model, B_model, alpha_model

def get_nnparams(nnmodels):
    # Description: A function to combine all the parameters of all nn models to be added to the optimizer later

    nn_params      = []
    for model in nnmodels:
        if model  != None:
            nn_params += list(model.parameters())
    
    return nn_params

