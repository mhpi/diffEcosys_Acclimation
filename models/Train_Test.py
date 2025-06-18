from utils.util               import process_range
from models.loss_function     import RangeBoundLoss
import torch
import torch.nn as nn


def NN_Train(param_list, nn_cat, nn_cont, nn_model,param_range_list):
    # Description: A function used to make nn predictions while training
    pred_list = []

    for n in range(len(param_list)):
        par = 0.0
        # for model in models_list:
        if nn_cat is None and nn_cont is not None:
            output = nn_model(nn_cont)
        elif nn_cat is not None and nn_cont is None:
            output = nn_model(nn_cat)
        else:
            output = nn_model(nn_cat, nn_cont)

        if len(param_list) == 1:
            par += output.reshape(-1) * (param_range_list[n][1] - param_range_list[n][0]) + param_range_list[n][0]
        else:
            par += output[:, n].reshape(-1) * (param_range_list[n][1] - param_range_list[n][0]) + param_range_list[n][0]

        pred_list.append(par)
    return pred_list
#==================================================================================================================
def Train_diffmodel( args, data_dict, models_dict, optim):
    # Description: A trainer function for the differentiable model (nn_models + physical models)
    # for each epoch:
        # 1. Run nn_models to make params predictions
        # 2. Run physical models to get simulations
        # 3. compute loss function
        # 4. Do the backward run to update the weights of the NNs

    # extract loss function
    criterion = nn.MSELoss()

    # extract both nn_models and physical models from models_dict
    V_model, B_model, alpha_model = models_dict['nn_models']
    photo_model, soil_model       = models_dict['physical_models']

    photo_model_data  = data_dict['physM_data']

    # extract target data, and input data to NNs
    targets                 = data_dict['target_data']
    varCat_NNv, varCont_NNv = data_dict['NNv_data']
    varCat_NNB, varCont_NNB = data_dict['NNB_data']

    params_all = {'Vcmax25':None, 'btran':None, 'g0':None, 'g1':None, "vcmaxha":None, "vcmaxse":None}
    param_list = args['param_list']
    param_range_list = args['param_range_list']
    len_alpha  = len(param_list)
    alpha      = [1.0] * len(param_list)

    # start the training epoch
    for i in range(args['epochs']):
        # Run all nn models to predict the parameters
        #======ALPHA MODEL==========
        if alpha_model is not None:
            AL_concat =torch.cat([varCat_NNv, varCont_NNv], dim=1)
            alpha = NN_Train(['alpha']*len_alpha,
                             nn_cat=None,
                             nn_cont=AL_concat,
                             nn_model=alpha_model,
                             param_range_list=[(0,1)]*len_alpha
                             )

        #======V MODEL==============
        if V_model is not None:
            predicted_params = NN_Train(param_list,
                                        nn_cat = varCat_NNv,
                                        nn_cont = None,
                                        nn_model = V_model,
                                        param_range_list = param_range_list
                                    )
            params_all.update(dict(zip(param_list, predicted_params)))
            for p_id in range(len(param_list)):
                p = param_list[p_id]
                try:
                    params_all[p] = torch.clamp(params_all[p] * alpha[p_id], min = param_range_list[p_id][0], max = param_range_list[p_id][1])
                except:
                    params_all[p] = torch.clamp(params_all[p] * alpha[p_id], min=param_range_list[0], max=param_range_list[1])

        idx         = data_dict['LGE_data'].shape[0]# photo_model_data['co2_ppress'].shape[0]
        params_diff = {key: value[:idx] if key in param_list and value is not None else value for key, value in params_all.items()}
        params_ML   = {key: value[idx:] if key in param_list and value is not None else value for key, value in params_all.items()}

        #======B MODEL==============
        if B_model is not None:
            B = NN_Train(['B'],
                         varCat_NNB.reshape(-1, 1),
                         varCont_NNB.reshape(-1,len(args['cont_cols_B'])),
                         B_model, [(0,1)])[0]

            btran_params         = soil_model.get_btran_params(data_dict['LGE_data'])
            params_diff['btran'] = soil_model.get_btran(B, btran_params)

        # Run the physical model: 1. solve, 2. get simulations for An, gs
        #======SOLVER===============
        if i == 0:
            ftol_epoch = 1000
            x_epoch    = photo_model_data['co2_ppress']
            x_epoch.requires_grad_(True)
        else:
            ftol_epoch = ftol.abs().max()
        # ======PHOTO MODEL===============
        photo_model.set_forc_attrs(data_dict['physM_data'])
        photo_model.set_params(params_diff)
        outputs, ftol, x_epoch = photo_model.predict_An_gs(ftol_epoch = ftol_epoch,
                                                           x_in       = x_epoch,
                                                           option     = args['solver'])

        subset_keys =  [item for item in args['target_cols'] if item not in args['var_cols']]
        outputs.update({key: params_ML[key] for key in  subset_keys})
        # compute the loss function
        # ======LOSS FUNCTI===============

        loss = torch.zeros_like(x_epoch.clone())[0]
        for ct, (key, value) in enumerate(targets.items()):
            if value != None:
                mask     = torch.where(~torch.isnan(value))
                obs_norm = (value[mask] - value[mask].mean())/value[mask].std()
                sim_norm = (outputs[f'{key}'][mask] - value[mask].mean())/value[mask].std()
                loss     += args['target_cols_w'][ct] * torch.sqrt(criterion(sim_norm, obs_norm))

        # Add to loss function a customized loss that penalizes parameters going outside a specified lower and upper bound
        # ======ALPHA MODEL===============
        if alpha_model is not None:
            penalty    = args['plty']
            loss_alpha = RangeBoundLoss([args['lb_alpha']]*len_alpha, [args['ub_alpha']]*len_alpha)
            loss       = loss + loss_alpha.forward(alpha, penalty)


        loss.backward()
        optim.step()
        photo_model.zero_grad(); soil_model.zero_grad()
        optim.zero_grad()
        print("Epoch = {}, loss = {:5.10f}, ftol = {:e}".format(i, loss, ftol.abs().max()))

    # after finishing all epochs saved trained_nn_models to be used for testing later
    models_dict['trained_nn_models'] = [[V_model.eval()]     if V_model is not None else [],
                                        [B_model.eval()]     if B_model is not None else [],
                                        [alpha_model.eval()] if alpha_model is not None else [],
                                     ]

    return models_dict
#=======================================================================================================================
#=======================================================================================================================
def NN_Test(nn_cat, nn_cont,trained_models , scaler = None, ):
    # Description: A function used to make nn predictions while testing
    models_list = trained_models

    all_preds = []

    # Process each model
    for model in models_list:
        if nn_cat is None and nn_cont is not None:
            output = model(nn_cont)
        elif nn_cat is not None and nn_cont is None:
            output = model(nn_cat)
        else:
            output = model(nn_cat, nn_cont)

        if scaler != None:
            scaled_output = output * scaler[1] + scaler[0]
        else:
            scaled_output = output

        # Stack parameter predictions for the current model
        all_preds.append(scaled_output)

    final_output = torch.stack(all_preds, dim=0)
    return final_output
#=======================================================================================================================
#=======================================================================================================================
def Test_diffmodel(args, data_dict, models_dict):
    # Description: A tester function for the differentiable model (nn_models + physical models)
    # 1. forward the pretrained nn_models to make params predictions
    # 2. Run physical models to get simulations
    # 3. save the simulations and parameters in the output dictionary to compute statistics against observations

    # extract the pretrained_nn_models
    V_models, B_models, alpha_models = models_dict['trained_nn_models']

    # extract the physical models
    photo_model, soil_model   = models_dict['physical_models']
    photo_model_data          = data_dict['physM_data']

    # extract the input data to NNs
    varCat_NNv, varCont_NNv = data_dict['NNv_data']
    varCat_NNB, varCont_NNB = data_dict['NNB_data']

    param_list = args['param_list']
    param_range_list = args['param_range_list']

    # forward the pre-trained nn models to predict the parameters
    with torch.no_grad():
        #======ALPHA MODEL==========
        alpha = 1.0
        if len(alpha_models) > 0:
            AL_concat =torch.cat([varCat_NNv, varCont_NNv], dim=1)
            alpha = NN_Test(nn_cat        = None,
                            nn_cont       = AL_concat,
                            trained_models= alpha_models)

        # ======V MODEL==============
        params_all    = {'Vcmax25': None, 'btran': None, 'g0': None, 'g1': None, "vcmaxha": None, "vcmaxse": None}

        if len(V_models)> 0:
            params_scaler = process_range(param_range_list, device=varCat_NNv.device)
            predicted_params =  NN_Test(nn_cat=varCat_NNv, nn_cont=None,scaler= params_scaler, trained_models=V_models)
            predicted_params = predicted_params * alpha
            predicted_params = torch.clamp(predicted_params, min = params_scaler[0], max = params_scaler[2]).mean(0)
            params_all.update(dict(zip(param_list, [predicted_params[:, i] for i in range(predicted_params.shape[1])])))

        idx         = photo_model_data['co2_ppress'].shape[0]
        params_diff = {key: value[:idx] if key in param_list and value is not None else value for key, value in params_all.items()}
        params_ML   = {key: value[idx:] if key in param_list and value is not None else value for key, value in params_all.items()}

        # ======B MODEL==============
    if len(B_models) > 0:
        with torch.no_grad():
            B = NN_Test(nn_cat=varCat_NNB.reshape(-1, 1),
                        nn_cont=varCont_NNB.reshape(-1, len(args['cont_cols_B'])),
                        trained_models= B_models).mean(0)[:,0]

        btran_params         = soil_model.get_btran_params(data_dict['LGE_data'])
        params_diff['btran'] = soil_model.get_btran(B, btran_params)
        B_lyrs = B.reshape(args['nly'], -1)
        params_diff['B0'] = B_lyrs[0]; params_diff['B1'] = B_lyrs[1]
        params_diff['B2'] = B_lyrs[2]; params_diff['B3'] = B_lyrs[3]
        params_diff['B4'] = B_lyrs[4]
    else:
        params_diff['btran'] = None

    # forward the physical model: 1. solve, 2. get simulations for An, gs
    # ======PHOTO MODEL===============
    photo_model.set_forc_attrs(data_dict['physM_data'])
    photo_model.set_params(params_diff)
    physical_outputs, ftol, x = photo_model.predict_An_gs(ftol_epoch  = 1000,
                                                          x_in        = photo_model_data['co2_ppress'],
                                                          option      = args['solver'])

    # save all required simulations or parameters in the output dictionary
    # ===================================
    for key, value in physical_outputs.items():
        physical_outputs[key] = value.detach().to('cpu').numpy()
    # ===================================
    param_outputs = params_diff.copy()
    for key, value in param_outputs.items():
        if value != None:
            param_outputs[key] = value.detach().to('cpu').numpy()

    outputs_other = params_ML.copy()
    for key, value in outputs_other.items():
        if value != None:
            outputs_other[key] = value.detach().to('cpu').numpy()

    outputs_LGE = dict()
    outputs_LGE.update(physical_outputs)
    outputs_LGE.update(param_outputs)
    outputs_LGE['ftol'] = ftol
    return outputs_LGE, outputs_other
#=======================================================================================================================

