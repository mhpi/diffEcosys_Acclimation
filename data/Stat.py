import numpy as np
import torch
import hydroeval as he
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler

def normalize(l, mean = None, std = None, to_norm = True):
    # Description: A function to normalize or denormalize array l using its mean and standard deviation
    if to_norm:
        " Normalize"
        if (mean != None) and (std != None):
            out = (l - mean)/std
        else:
            out = (l - l.mean())/l.std()
    else:
        " DeNormalize"
        if (mean != None) and (std != None):
            out = l * std + mean
        else:
            raise RuntimeError("Missing inputs required to de-normalize")

    return out

def standize(l, min = None, max = None, to_stand = True):
    # Description: A function to standardize or de-standardize array l using maximum and minimum values
    if to_stand:
        " standardize"
        if (min != None) and (max != None):
            out = (l - min)/(max- min)
        else:
            out = (l - l.min())/(l.max() - l.min())
    else:
        " De-standardize"
        if (min != None) and (max != None):
            out = l * (max - min) + min
        else:
            raise RuntimeError("Missing inputs required to de-standardize")

    return out

def nse(predictions, targets):
    # Description: A function to compute the Nash Sutcliffe Efficiency (NSE) using predictions and targets
    try:
        return (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))
    except:
        return (1-torch.sum((predictions - targets) ** 2) / torch.sum((targets - torch.mean(targets)) ** 2))

def create_stat_dict(varLST, x, mtd=0):

    # Description: A function that computes statistical summaries (mean, std, percentiles) for each variable.
    #
    # Inputs
    # varLST : List of variable names corresponding to each column in `x`.
    # x : 2D array of shape (n_samples, n_features), where each column corresponds to a variable in varLST.
    # mtd : Method used to compute the statistics, int, optional (default=0)
         #  0 : Standard mean and std over the full data range, along with 5th and 95th percentiles.
         # 1 : Mean and std computed only from values between 5th and 95th percentiles (percentile-based trimming).
         # 2 : Robust scaling using median and IQR via sklearn's RobustScaler; uses 25th and 75th percentiles.

    # Outputs
    # stat_dict : a dictionary mapping variable names to their [mean, std, lower_bound, upper_bound], where:
    #     - mean : average value or robust mean
    #     - std : standard deviation or robust std
    #     - lower_bound : 5th percentile (methods 0 & 1) or 25th percentile (method 2)
    #     - upper_bound : 95th percentile (methods 0 & 1) or 75th percentile (method 2)

    stat_lst = []
    vars_temp = x.copy()
    for k in range(len(varLST)):

        if mtd == 0:
            # Calculate the 5th and 95th percentiles of one column
            lower_bound, upper_bound = np.percentile(vars_temp[:,k], [5, 95])
            # Normalization without percentiles
            mean = vars_temp[:,k].mean()
            std  = vars_temp[:,k].std()
            stat_lst.append([mean, std, lower_bound, upper_bound])

        elif mtd == 1:
            # Calculate the 5th and 95th percentiles of one column
            lower_bound, upper_bound = np.percentile(vars_temp[:,k], [5, 95])
            # Select values within the 5th and 95th percentiles
            values_within_bounds = vars_temp[:,k][(vars_temp[:,k] >= lower_bound) & (vars_temp[:,k] <= upper_bound)]
            # Calculate mean and standard deviation of the values within bounds
            stat_lst.append([values_within_bounds.mean(), values_within_bounds.std(), lower_bound, upper_bound])

        elif mtd == 2:
            # Robust scaling
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(vars_temp[:,k].reshape(-1, 1))
            mean = scaled_data.mean()
            std  = scaled_data.std()
            # Bounds are not defined for robust scaling in the same way, use quartiles as a reference
            lower_bound, upper_bound = np.percentile(scaled_data, [25, 75])
            stat_lst.append([mean, std, lower_bound, upper_bound])

    stat_dict = dict(zip(varLST, stat_lst))
    return stat_dict


def cal_stats(pred, obs):
    # Description: A function to create dictionary that has different statistical metrics such as
    # COR : Pearson's Correlation Coefficient
    # RMSE: Root mean square error
    # BIAS: Bias
    # NSE : Nash Sutcliffe Efficiency

    # Inputs:
    # pred  : Predictions
    # obs   : Observations

    # Outputs:
    # stats  : dictionary with the statistical metrics

    COR , _ = pearsonr(pred, obs)
    RMS  = np.sqrt(np.nanmean((pred- obs) ** 2))
    Bias = np.nanmean(pred- obs)
    NSE  = nse(pred, obs)
    kge, r, alpha, beta = he.evaluator(he.kge, pred, obs)
    stats = {'COR': COR, 'RMS': RMS, 'Bias': Bias, 'NSE': NSE, 'KGE':kge[0]}
    return stats

def scale(args, x, mtd, function = "norm"):
    
    # Description: A function that scales continuous variables using specified normalization method.

    # Inputs:
    # args: arguments from config file
    # x   : 2D array of input data with shape (n_samples, n_features), corresponding to cont_cols_v.
    # mtd : Method to choose which statistical dictionary to use:
             # 0 : Create a new statistics dictionary using create_stat_dict, usually when training
             # 1 : Use args[exp_dict], usually when testing to use the same stats as used for training
             # else : Use args[stat_dict], a presaved dictionary of statisitcs, used for global runs
     
    # function : str or None, default = "norm"
             #  norm : Normalize using mean and standard deviation (z-score normalization)
             #  stand: Standardize using minimum and maximum
             #  None : No scaling applied
    
    # Outputs:
    # out : scaled data array of same shape as x
    
    varLst = args['cont_cols_v']
    if mtd == 0: #All
        dict = create_stat_dict(varLst, x)
        args['exp_dict'] = dict
    elif mtd ==1:
        dict = args['exp_dict']
    else:
        dict = args['stat_dict']

    out       = np.zeros(x.shape)
    vars_temp = x.copy()
    for k in range(len(varLst)):
        var  = varLst[k]
        stat = dict[var]
        if function == "norm":
            assert len(x.shape) == 2
            out[:, k] = normalize(vars_temp[:, k] , mean = stat[0], std = stat[1])
        elif function == "stand":
            out[:, k] = standize(vars_temp[:, k])
        elif function == None:
            out[:, k] = var
        else:
            print("scaling function is not defined")

    return out





