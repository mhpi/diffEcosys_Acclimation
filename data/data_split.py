import pandas as pd
import numpy as np

#######################################################################################################################
def create_sets_temporal(df_all, ratio):
    # Description: A function to split the dataset temporally with train = ratio * size of dataset
    # Inputs:
    # df_all: The dataframe dataset
    # ratio : percentage of the time used for training

    df_all['Date'] = pd.to_datetime(df_all['Date'])
    df_all         = df_all.sort_values(by = ['Location','Date'])
    df_all         = df_all.reset_index(drop = True)

    train_set = pd.DataFrame()
    test_set  = pd.DataFrame()

    # First location layer
    for loc in df_all.Location.unique():
        df_loc = df_all[df_all['Location'] == loc]

        # Second PFT Layer
        for pft in df_loc.PFT.unique():
            df_temp = df_loc[df_loc['PFT'] == pft]

            # Number of dates of measurements available for a specific PFT in a specific location
            n       = len(df_temp['Date'].unique())
            if n == 1:
                train_set= pd.concat([train_set,df_temp])
            else:
                ntrain    = np.int(np.floor(ratio * n))
                All_dates = np.sort(df_temp['Date'].unique())
                train_date= All_dates[:ntrain]
                df_train  = df_temp[df_temp['Date'].isin(train_date)]
                df_test   = df_temp.drop(df_train.index)
                train_set = pd.concat([train_set,df_train])
                test_set  = pd.concat([test_set ,df_test])

    train_set = train_set.reset_index(drop=True)
    test_set  = test_set.reset_index(drop=True)

    return train_set, test_set

#######################################################################################################################
