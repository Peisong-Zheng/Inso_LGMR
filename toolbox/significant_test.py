import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyEDM import CCM
from scipy.stats import zscore

def ccm_significance_statistic(ds_sat, df_pre, ds_sat_ens, lat_idx, lon_idx, ens_sample, 
                     E_val, tau_val, Tp_val, libSizes, sample=10, show_figures=True):
    """
    Run CCM analysis at a specified grid point using the mean SAT, ensemble SAT, 
    and an interpolated version of pre with random age shifts.
    
    Parameters:
        ds_sat      : xarray.Dataset 
                      Mean SAT dataset (must include 'sat', 'age', 'lat', and 'lon').
        df_pre      : pandas.DataFrame 
                      Pre dataset (must include 'pre' and 'age' columns).
        ds_sat_ens  : xarray.Dataset 
                      Ensemble SAT dataset (must include 'sat', 'lat', 'lon', and 'age').
        lat_idx     : int
                      Index for the latitude grid.
        lon_idx     : int
                      Index for the longitude grid.
        samples     : int
                      Number of ensemble members (and random age series) to use.
        E_val       : int
                      Embedding dimension.
        tau_val     : int
                      Time delay.
        Tp_val      : int
                      Prediction horizon.
        libSizes    : str
                      Library sizes for CCM (e.g. "10 20 30 40 50 60 70").
        show_figures: bool, optional
                      If True, display intermediate plots (default is True).
    
    Returns:
        dict : A dictionary with the following keys:
            "ccm_mean"         - CCM result (DataFrame) from the mean SAT.
            "ensemble_ccm"     - List of CCM results (DataFrames) from the shifted ensemble.
            "sat_mean"         - 1D array of the mean SAT time series.
            "sat_ens_shifted"  - 2D array (samples x time) of shifted ensemble SAT data.
            "pre_ran"          - 2D array (samples x time) of interpolated pre data.
            "time"             - 1D array of the time coordinate.
    """
    # ---------------------------
    # 1. Data extraction and sampling
    # ---------------------------
    time = ds_sat['age'].values
    sat_mean = ds_sat['sat'].isel(lat=lat_idx, lon=lon_idx).values
    sat_ens = ds_sat_ens['sat'].isel(lat=lat_idx, lon=lon_idx).values

  

    # df_pre['age'] = df_pre['age'] 
    # df_pre['pre'] = df_pre['pre'].values

    # # flip the time order of the ds_sat['sat']
    # sat_mean = sat_mean[::-1]
    # sat_ens = sat_ens[:, ::-1]
    # df_pre = df_pre[::-1]



    
    # Randomly select "samples" ensemble members
    sat_ens = sat_ens[np.random.choice(sat_ens.shape[0], ens_sample, replace=False), :]
    
    # ---------------------------
    # 2. Generate random age series and interpolate pre for each sample
    # ---------------------------
    nTime = len(time)
    # For each time point, generate a random integer between (time[i]-100) and (time[i]+100)
    sat_age_ran = np.empty((ens_sample, nTime))
    for i in range(ens_sample):
        # np.random.randint can work with arrays if low and high are arrays
        sat_age_ran[i] = np.random.randint(time - 100, time + 99)
    
    pre_arr = df_pre['pre'].values
    pre_age = df_pre['age'].values 
    pre_ran = np.empty((ens_sample, len(pre_age)))
    for i in range(ens_sample):
        pre_ran[i] = np.interp(sat_age_ran[i], pre_age, pre_arr)
    
    # ---------------------------
    # 3. Plot Mean SAT and ensemble members (original)
    # ---------------------------
    if show_figures:
        plt.figure(figsize=(10, 5))
        for i in range(sat_ens.shape[0]):
            plt.plot(time, zscore(sat_ens[i, :]), color='gray', alpha=0.3)
        plt.plot(time, zscore(sat_mean), color='k', lw=2, label='Mean SAT')
        plt.plot(df_pre['age'], zscore(df_pre['pre']), color='b', lw=2, label='Pre')
        plt.xlabel("Time (age)")
        plt.ylabel("SAT")
        plt.title(f"Mean SAT vs. Ensemble SAT at lat={int(ds_sat['lat'].values[lat_idx])}, lon={ds_sat['lon'].values[lon_idx]}")
        plt.legend()
        plt.show()
    
    # ---------------------------
    # 4. Create shifted ensemble data by breaking and swapping halves
    # ---------------------------
    half = nTime // 2
    sat_ens_shifted = np.empty_like(sat_ens)
    for i in range(sat_ens.shape[0]):
        ts = sat_ens[i, :]
        shifted_ts = np.concatenate((ts[half:], ts[:half]))
        sat_ens_shifted[i, :] = shifted_ts
    
    if show_figures:
        plt.figure(figsize=(10, 5))
        for i in range(sat_ens_shifted.shape[0]):
            plt.plot(time, sat_ens_shifted[i, :], color='orange', alpha=0.2)
        plt.plot(time, sat_mean, color='k', lw=2, label='Mean SAT')
        plt.xlabel("Time (age)")
        plt.ylabel("SAT")
        plt.title("Mean SAT vs. Shifted Ensemble SAT")
        plt.legend()
        plt.show()
    
    # ---------------------------
    # 5. CCM analysis using the mean SAT and pre data
    # ---------------------------
    df_ccm = pd.DataFrame({
        'Time': time,
        'X': sat_mean,
        'Y': df_pre['pre'].values
    })
    ccm_out = CCM(
        dataFrame   = df_ccm,
        E           = E_val,
        tau         = tau_val,
        columns     = "X",   # SAT manifold
        target      = "Y",   # predict pre
        libSizes    = libSizes,
        sample      = sample,
        random      = True,
        replacement = False,
        Tp          = Tp_val
    )
    
    # ---------------------------
    # 6. CCM analysis for each shifted ensemble member using its corresponding interpolated pre
    # ---------------------------
    ensemble_ccm = []
    for i in range(sat_ens_shifted.shape[0]):
        df_temp = pd.DataFrame({
            'Time': time,
            'X': sat_ens_shifted[i, :],
            'Y': pre_ran[i]
        })
        try:
            out = CCM(
                dataFrame   = df_temp,
                E           = E_val,
                tau         = tau_val,
                columns     = "X",
                target      = "Y",
                libSizes    = libSizes,
                sample      = sample,
                random      = True,
                replacement = False,
                Tp          = Tp_val
            )
            ensemble_ccm.append(out)
        except Exception as e:
            print(f"Error in ensemble member {i}: {e}")
    
    # ---------------------------
    # 7. Plot the CCM curves in a subplot with two panels
    # ---------------------------
    if show_figures:
        fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
        
        # Left subplot: SAT -> pre (using "X:Y")
        ax = axes[0]
        for i, out in enumerate(ensemble_ccm):
            label = 'Ensemble SAT CCM' if i == 0 else None
            ax.plot(out['LibSize'], out['X:Y'], color='lightcoral', linestyle='-', alpha=0.3, label=label)
        ax.plot(ccm_out['LibSize'], ccm_out['X:Y'], 'ro-', label='Mean SAT CCM')
        ax.set_xlabel("Library Size")
        ax.set_ylabel("Prediction Skill (rho)")
        ax.set_title(r'$\hat{pre}|M_{sat}$')
        
        # Right subplot: pre -> SAT (using "Y:X")
        ax2 = axes[1]
        for i, out in enumerate(ensemble_ccm):
            label = 'Ensemble pre CCM' if i == 0 else None
            ax2.plot(out['LibSize'], out['Y:X'], color='skyblue', linestyle='-', alpha=0.1, label=label)
        ax2.plot(ccm_out['LibSize'], ccm_out['Y:X'], 'bo-', label='Mean pre CCM')
        ax2.set_xlabel("Library Size")
        ax2.set_ylabel("Prediction Skill (rho)")
        ax2.set_title(r'$\hat{sat}|M_{pre}$')
        
        plt.tight_layout()
        plt.show()
    
    # ---------------------------
    # 8. Return results
    # ---------------------------
    return {
        "ccm_mean": ccm_out,
        "ensemble_ccm": ensemble_ccm,
        "sat_mean": sat_mean,
        "sat_ens_shifted": sat_ens_shifted,
        "pre_ran": pre_ran,
        "time": time
    }


def ccm_significance_test(ccm_mean, ensemble_ccm, uni_dir=False, if_plot=False):
    """
    Test whether the CCM result for the mean is significantly different from that of the shifted ensemble.
    
    Parameters:
      ccm_mean : pandas.DataFrame
          CCM output for the mean data. Must contain columns "LibSize", "X:Y", and "Y:X".
      ensemble_ccm : list of pandas.DataFrame
          A list of CCM outputs for each ensemble member, with the same columns as ccm_mean.
          
    Returns:
      bool: True if the CCM using SAT to predict pre is significantly different 
            (i.e. the mean value is outside the 5th-95th percentile of the ensemble) 
            AND the CCM using pre to predict SAT is not significant (i.e. the mean falls 
            within the ensemble range). Returns False otherwise.
    """
    # Use the maximum LibSize as the test point.
    max_lib = ccm_mean["LibSize"].max()
    

    mean_sat2pre = np.mean(ccm_mean['X:Y'])
    mean_pre2sat = np.mean(ccm_mean['Y:X'])
    
    # Gather ensemble values at the maximum LibSize.
    ens_sat2pre = []
    ens_pre2sat = []
    for ens_df in ensemble_ccm:
        try:
            # val_sat2pre = ens_df.loc[ens_df["LibSize"] == max_lib, "X:Y"].values[0]
            # val_pre2sat = ens_df.loc[ens_df["LibSize"] == max_lib, "Y:X"].values[0]
            val_sat2pre = np.mean(ens_df['X:Y'])
            val_pre2sat = np.mean(ens_df['Y:X'])
            ens_sat2pre.append(val_sat2pre)
            ens_pre2sat.append(val_pre2sat)
        except Exception as e:
            print(f"Error extracting ensemble data: {e}")
    
    ens_sat2pre = np.array(ens_sat2pre)
    ens_pre2sat = np.array(ens_pre2sat)

    if if_plot:
        # in case uni_dir is false plot figure with two subplots
    
        if uni_dir:
            # plot the histogram of the ensemble values and a vertical line for the mean
            fig, ax = plt.subplots(1, 1, figsize=(6, 4),dpi=100)
            ax.hist(ens_sat2pre, bins=20, density=True, color='lightcoral', alpha=0.5, label='Ensemble SAT->pre')
            ax.axvline(mean_sat2pre, color='red', linestyle='--', label='Mean SAT->pre')
            ax.set_title(r'$\hat{pre}|M_{sat}$')
            # add x-axis label
            ax.set_xlabel("Prediction Skill (ρ)")
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.show()
        else:
            # plot the histogram of the ensemble values and a vertical line for the mean
            fig, axes = plt.subplots(1, 2, figsize=(12, 4),dpi=100)
            ax1 = axes[0]
            ax2 = axes[1]
            ax1.hist(ens_sat2pre, bins=20, density=True, color='lightcoral', alpha=0.5, label='Ensemble SAT->pre')
            ax1.axvline(mean_sat2pre, color='red', linestyle='--', label='Mean SAT->pre')
            ax1.set_title(r'$\hat{pre}|M_{sat}$')
            # add x-axis label
            ax1.set_xlabel("Prediction Skill (ρ)")
            ax1.set_ylabel("Frequency")
    

            # ax1.legend()
            ax2.hist(ens_pre2sat, bins=20, density=True, color='skyblue', alpha=0.5, label='Ensemble pre->SAT')
            ax2.axvline(mean_pre2sat, color='blue', linestyle='--', label='Mean pre->SAT')
            ax2.set_title(r'$\hat{sat}|M_{pre}$')
            # add x-axis label
            ax2.set_xlabel("Prediction Skill (ρ)")
            ax2.set_ylabel("Frequency")


        # ax2.legend()
        plt.show()
    
    # Compute the 5th and 95th percentiles of the ensemble distributions.
    lower_sat2pre = np.percentile(ens_sat2pre, 5)
    upper_sat2pre = np.percentile(ens_sat2pre, 95)
    lower_pre2sat = np.percentile(ens_pre2sat, 5)
    upper_pre2sat = np.percentile(ens_pre2sat, 95)
    
    # Condition 1: Mean SAT->pre prediction (X:Y) is outside the ensemble range.
    significant_sat2pre = (mean_sat2pre > upper_sat2pre)
    
    # Condition 2: Mean pre->SAT prediction (Y:X) is within the ensemble range.
    non_significant_pre2sat = (mean_pre2sat <= upper_pre2sat)
    
    # return significant_sat2pre and non_significant_pre2sat
    if uni_dir:
        return significant_sat2pre
    else:
        return significant_sat2pre and non_significant_pre2sat



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyEDM import CCM










def ccm_significance_test_v1(ds_sat, df_pre, ds_sat_ens, lat_idx, lon_idx,
    E=4, 
    tau=8, 
    n_ran=20, 
    libSizes="100 200 300 400 500 600 700",
    Tp=0,
    sample=100,
    showPlot=True
):


    time = ds_sat['age'].values
    sat_mean = ds_sat['sat'].isel(lat=lat_idx, lon=lon_idx).values
    sat_ens = ds_sat_ens['sat'].isel(lat=lat_idx, lon=lon_idx).values



    
    # Randomly select "samples" ensemble members
    sat_ens = sat_ens[np.random.choice(sat_ens.shape[0], n_ran, replace=False), :]
    
    # ---------------------------
    # 2. Generate random age series and interpolate pre for each sample
    # ---------------------------
    nTime = len(time)
    # For each time point, generate a random integer between (time[i]-100) and (time[i]+100)
    sat_age_ran = np.empty((n_ran, nTime))
    for i in range(n_ran):
        # np.random.randint can work with arrays if low and high are arrays
        sat_age_ran[i] = np.random.randint(time - 100, time + 99)
    
    pre_arr = df_pre['pre'].values
    pre_age = df_pre['age'].values 
    pre_ran = np.empty((n_ran, len(pre_age)))
    for i in range(n_ran):
        pre_ran[i] = np.interp(sat_age_ran[i], pre_age, pre_arr)
    
    # ---------------------------
    # 3. Plot Mean SAT and ensemble members (original)
    # ---------------------------
    if showPlot:
        plt.figure(figsize=(10, 5))
        for i in range(sat_ens.shape[0]):
            plt.plot(time, zscore(sat_ens[i, :]), color='gray', alpha=0.3)
        plt.plot(time, zscore(sat_mean), color='k', lw=2, label='Mean SAT')
        plt.plot(df_pre['age'], zscore(df_pre['pre']), color='b', lw=2, label='Pre')
        plt.xlabel("Time (age)")
        plt.ylabel("SAT")
        plt.title(f"Mean SAT vs. Ensemble SAT at lat={int(ds_sat['lat'].values[lat_idx])}, lon={ds_sat['lon'].values[lon_idx]}")
        plt.legend()
        plt.show()

    # half = nTime // 2
    sat_ens_shifted = np.empty_like(sat_ens)
    for i in range(sat_ens.shape[0]):
        ts = sat_ens[i, :]
        break_point = np.random.randint(len(ts)//5, len(ts)*4//5)
        randomized_swapped = np.concatenate([ts[break_point:], ts[:break_point]])
        # shifted_ts = np.concatenate((ts[half:], ts[:half]))
        sat_ens_shifted[i, :] = randomized_swapped



    if showPlot:
        plt.figure(figsize=(10, 5))
        for i in range(sat_ens_shifted.shape[0]):
            plt.plot(time, sat_ens_shifted[i, :], color='orange', alpha=0.2)
        plt.plot(time, sat_mean, color='k', lw=2, label='Mean SAT')
        plt.xlabel("Time (age)")
        plt.ylabel("SAT")
        plt.title("Mean SAT vs. Shifted Ensemble SAT")
        plt.legend()
        plt.show()
    

    df = pd.DataFrame({
        "Time": df_pre["age"],
        "X":    sat_mean,
        "Y":    df_pre[df_pre.columns[1]]
    })

    column_name='sat'
    target_name=df_pre.columns[1]


    # Real-data CCM
    ccm_out = CCM(
        dataFrame   = df,
        E           = E,
        tau         = tau,
        columns     = "X",   # predictor
        target      = "Y",   # target
        libSizes    = libSizes,
        sample      = sample,
        random      = True,
        replacement = False,
        Tp          = Tp
    )


    ran_ccm_list_xy = []
    for i in range(sat_ens_shifted.shape[0]):
        df_temp = pd.DataFrame({
            'Time': time,
            'X': sat_ens_shifted[i, :],
            'Y': pre_ran[i]
        })
        try:
            out = CCM(
                dataFrame   = df_temp,
                E           = E,
                tau         = tau,
                columns     = "X",
                target      = "Y",
                libSizes    = libSizes,
                sample      = sample,
                random      = True,
                replacement = False,
                Tp          = Tp
            )
            ran_ccm_list_xy.append(out)
        except Exception as e:
            print(f"Error in ensemble member {i}: {e}")


    if showPlot:
        # create a figure and plot the original time series and the randomized time series
        fig1, ax = plt.subplots(1, 1, figsize=(10, 3),dpi=100)
        ax.plot(df["Time"], df["X"], label=column_name)
        # ax.plot(df["Time"], df["Y"], label=target_name)
        # plot the randomized time series
        for i in range(n_ran):
            ax.plot(df["Time"], sat_ens_shifted[i,:], color='grey', alpha=0.3)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()

    # Optionally plot results
    if showPlot:

        fig, ax = plt.subplots(figsize=(4, 4))

        libsize = ran_ccm_list_xy[0]["LibSize"].values


        yx_surrogates = np.column_stack([out_xy["Y:X"].values for out_xy in ran_ccm_list_xy])
        # yx_min = yx_surrogates.min(axis=1)
        # let the yx_min to be the 5th percentile of the yx_surrogates
        yx_min = np.percentile(yx_surrogates, 5, axis=1)
        # yx_max = yx_surrogates.max(axis=1)
        # let the yx_max to be the 95th percentile of the yx_surrogates
        yx_max = np.percentile(yx_surrogates, 95, axis=1)

        xy_surrogates = np.column_stack([out_xy["X:Y"].values for out_xy in ran_ccm_list_xy])
        # xy_min = xy_surrogates.min(axis=1)
        # xy_max = xy_surrogates.max(axis=1)
        # let the xy_min to be the 5th percentile of the xy_surrogates
        xy_min = np.percentile(xy_surrogates, 5, axis=1)
        # let the xy_max to be the 95th percentile of the xy_surrogates
        xy_max = np.percentile(xy_surrogates, 95, axis=1)

        # Fill between min and max for X->Y
        ax.fill_between(libsize, xy_min, xy_max, color="r", alpha=0.2, label='', edgecolor='none')

        # Fill between min and max for Y->X
        ax.fill_between(libsize, yx_min, yx_max, color="b", alpha=0.2, label='', edgecolor='none')


        ax.plot(ccm_out["LibSize"], ccm_out["Y:X"], "b-",
                label=fr"$\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")

        ax.plot(ccm_out["LibSize"], ccm_out["X:Y"], "r-",
                label=fr"$\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
        
        # set the xlim to match the range of the libsize
        ax.set_xlim([libsize[0], libsize[-1]])

        # set ylim to be -0.1 to 1.1
        ax.set_ylim([-0.15, 1.15])

        ax.set_xlabel("Library Size")
        ax.set_ylabel("Prediction Skill (rho)")
        ax.legend()
        plt.tight_layout()
        plt.show()



    test_result=ccm_significance_hist(ccm_out, ran_ccm_list_xy, column_name=column_name, target_name=target_name, if_plot=showPlot)

    return ccm_out, ran_ccm_list_xy, test_result







def ccm_significance_test_v2(
    df_sd, 
    df_pre,
    E=4, 
    tau=8, 
    n_ran=20, 
    libSizes="100 200 300 400 500 600 700",
    Tp=0,
    sample=100,
    showPlot=True
):
    """
    Perform a CCM significance test by:
      1) Building a DataFrame with X, Y from df_sd and df_pre.
      2) Running CCM on the real data.
      3) Generating 'n_ran' surrogate versions of X (with random perturbations),
         each time re-running CCM, storing results in ran_ccm_list_xy.
      4) Optionally plotting real vs. surrogate cross mappings.

    Parameters
    ----------
    df_sd : pd.DataFrame
        DataFrame containing at least ["age"] and one data column for X.
    df_pre : pd.DataFrame
        DataFrame containing at least ["age"] and one data column for Y.
    E : int
        Embedding dimension (default=4).
    tau : int
        Time delay (default=8).

    n_ran : int
        Number of surrogate draws (default=20).
    libSizes : str or list
        Library sizes for CCM (default="100 200 300 400 500 600 700").
    sample : int
        Number of bootstrap samples in each CCM call (default=100).
    showPlot : bool
        Whether to show the resulting figure (default=True).

    Returns
    -------
    ccm_out : pd.DataFrame
        CCM output for the real data, containing columns like ["LibSize", "X:Y", "Y:X"].
    ran_ccm_list_xy : list
        List of CCM outputs (DataFrames) from each of the n_ran surrogate runs.
    """

    def randomize_stadial(stadial_data, seed=None):
        """
        1) Multiply original data by (1 + random variation in [-fraction, fraction]).
        2) Chop in half and rejoin (destroys original time ordering).
        """
        if seed is not None:
            np.random.seed(seed)
        
        # randomly select a break point between 2/10 and 8/10 of the data
        break_point = np.random.randint(len(stadial_data)//5, len(stadial_data)*4//5)
        randomized_swapped = np.concatenate([stadial_data[break_point:], stadial_data[:break_point]])
        
        return randomized_swapped

    # Build combined DataFrame: time, X, Y
    # We use the second column in df_sd and df_pre as X and Y, respectively.
    df = pd.DataFrame({
        "Time": df_pre["age"],
        "X":    df_sd[df_sd.columns[1]],
        "Y":    df_pre[df_pre.columns[1]]
    })

    column_name=df_sd.columns[1]
    target_name=df_pre.columns[1]


    # Real-data CCM
    ccm_out = CCM(
        dataFrame   = df,
        E           = E,
        tau         = tau,
        columns     = "X",   # predictor
        target      = "Y",   # target
        libSizes    = libSizes,
        sample      = sample,
        random      = True,
        replacement = False,
        Tp          = Tp
    )

    # create an array to store the randomly generated time X time series
    ran_time_series = np.zeros((n_ran, len(df["X"])))
    # Generate surrogate draws
    ran_ccm_list_xy = []
    for i in range(n_ran):
        # 1) Generate random surrogate for X
        X_ran = randomize_stadial(df["X"].values)
        # add the randomized time series to the array
        ran_time_series[i] = X_ran

        
        # 2) Create DataFrame with the same Y but newly randomized X
        df_surr = pd.DataFrame({
            "Time": df["Time"],
            "X":    X_ran,
            "Y":    df["Y"].values
        })
        
        # 3) Run CCM for X->Y on the surrogate data
        out_xy = CCM(
            dataFrame   = df_surr,
            E           = E,
            tau         = tau,
            columns     = "X",
            target      = "Y",
            libSizes    = libSizes,
            sample      = sample,
            random      = True,
            replacement = False,
            Tp          = Tp
        )
        ran_ccm_list_xy.append(out_xy)

    if showPlot:
        # create a figure and plot the original time series and the randomized time series
        fig1, ax = plt.subplots(1, 1, figsize=(10, 3),dpi=100)
        ax.plot(df["Time"], zscore(df["X"]), label=column_name)
        ax.plot(df["Time"], zscore(df["Y"]), label=target_name)
        # plot the randomized time series
        for i in range(n_ran):
            ax.plot(df["Time"], zscore(ran_time_series[i]), color='grey', alpha=0.3)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()

    # Optionally plot results
    if showPlot:

        fig, ax = plt.subplots(figsize=(4, 4))

        libsize = ran_ccm_list_xy[0]["LibSize"].values


        yx_surrogates = np.column_stack([out_xy["Y:X"].values for out_xy in ran_ccm_list_xy])
        # yx_min = yx_surrogates.min(axis=1)
        # let the yx_min to be the 5th percentile of the yx_surrogates
        yx_min = np.percentile(yx_surrogates, 5, axis=1)
        # yx_max = yx_surrogates.max(axis=1)
        # let the yx_max to be the 95th percentile of the yx_surrogates
        yx_max = np.percentile(yx_surrogates, 95, axis=1)

        xy_surrogates = np.column_stack([out_xy["X:Y"].values for out_xy in ran_ccm_list_xy])
        # xy_min = xy_surrogates.min(axis=1)
        # xy_max = xy_surrogates.max(axis=1)
        # let the xy_min to be the 5th percentile of the xy_surrogates
        xy_min = np.percentile(xy_surrogates, 5, axis=1)
        # let the xy_max to be the 95th percentile of the xy_surrogates
        xy_max = np.percentile(xy_surrogates, 95, axis=1)

        # Fill between min and max for X->Y
        ax.fill_between(libsize, xy_min, xy_max, color="r", alpha=0.2, label='', edgecolor='none')

        # Fill between min and max for Y->X
        ax.fill_between(libsize, yx_min, yx_max, color="b", alpha=0.2, label='', edgecolor='none')


        ax.plot(ccm_out["LibSize"], ccm_out["Y:X"], "b-",
                label=fr"$\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")

        ax.plot(ccm_out["LibSize"], ccm_out["X:Y"], "r-",
                label=fr"$\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
        
        # set the xlim to match the range of the libsize
        ax.set_xlim([libsize[0], libsize[-1]])

        # set ylim to be -0.1 to 1.1
        ax.set_ylim([-0.15, 1.15])

        ax.set_xlabel("Library Size")
        ax.set_ylabel("Prediction Skill (rho)")
        ax.legend()
        plt.tight_layout()
        plt.show()



    test_result=ccm_significance_hist(ccm_out, ran_ccm_list_xy, column_name=column_name, target_name=target_name, if_plot=showPlot)

    return ccm_out, ran_ccm_list_xy, test_result





def ccm_significance_hist(ccm_mean, ensemble_ccm, column_name='sat', target_name='pre', if_plot=False):
    """
    Test whether the CCM result for the mean is significantly different from that of the shifted ensemble.
    
    Parameters:
      ccm_mean : pandas.DataFrame
          CCM output for the mean data. Must contain columns "LibSize", "X:Y", and "Y:X".
      ensemble_ccm : list of pandas.DataFrame
          A list of CCM outputs for each ensemble member, with the same columns as ccm_mean.
          
    Returns:
      bool: True if the CCM using SAT to predict pre is significantly different 
            (i.e. the mean value is outside the 5th-95th percentile of the ensemble) 
            AND the CCM using pre to predict SAT is not significant (i.e. the mean falls 
            within the ensemble range). Returns False otherwise.
    """
    # Use the maximum LibSize as the test point.
    max_lib = ccm_mean["LibSize"].max()
    

    mean_sat2pre = np.mean(ccm_mean['X:Y'])
    mean_pre2sat = np.mean(ccm_mean['Y:X'])
    
    # Gather ensemble values at the maximum LibSize.
    ens_sat2pre = []
    ens_pre2sat = []
    for ens_df in ensemble_ccm:
        try:
            # val_sat2pre = ens_df.loc[ens_df["LibSize"] == max_lib, "X:Y"].values[0]
            # val_pre2sat = ens_df.loc[ens_df["LibSize"] == max_lib, "Y:X"].values[0]
            val_sat2pre = np.mean(ens_df['X:Y'])
            val_pre2sat = np.mean(ens_df['Y:X'])
            ens_sat2pre.append(val_sat2pre)
            ens_pre2sat.append(val_pre2sat)
        except Exception as e:
            print(f"Error extracting ensemble data: {e}")
    
    ens_sat2pre = np.array(ens_sat2pre)
    ens_pre2sat = np.array(ens_pre2sat)

    if if_plot:

            # plot the histogram of the ensemble values and a vertical line for the mean
            fig, axes = plt.subplots(1, 2, figsize=(12, 4),dpi=100)
            ax1 = axes[0]
            ax2 = axes[1]
            ax1.hist(ens_sat2pre, bins=20, density=True, color='lightcoral', alpha=0.5, label=fr"H0 $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
            ax1.axvline(mean_sat2pre, color='red', linestyle='--', label=fr"Real $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
            # ax1.set_title(r'$\hat{pre}|M_{sat}$')
            # add x-axis label
            ax1.set_xlabel("Prediction Skill (ρ)")
            ax1.set_ylabel("Frequency")
    

            # ax1.legend()
            ax2.hist(ens_pre2sat, bins=20, density=True, color='skyblue', alpha=0.5, label=fr"H0 $\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")
            ax2.axvline(mean_pre2sat, color='blue', linestyle='--', label=fr"Real $\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")
            # ax2.set_title(r'$\hat{sat}|M_{pre}$')
            # add x-axis label
            ax2.set_xlabel("Prediction Skill (ρ)")
            ax2.set_ylabel("Frequency")


            # ax2.legend()
            plt.show()
    
    # Compute the 5th and 95th percentiles of the ensemble distributions.
    lower_sat2pre = np.percentile(ens_sat2pre, 5)
    upper_sat2pre = np.percentile(ens_sat2pre, 95)
    lower_pre2sat = np.percentile(ens_pre2sat, 5)
    upper_pre2sat = np.percentile(ens_pre2sat, 95)
    
    # Condition 1: Mean SAT->pre prediction (X:Y) is outside the ensemble range.
    significant_sat2pre = (mean_sat2pre > upper_sat2pre)
    
    # Condition 2: Mean pre->SAT prediction (Y:X) is within the ensemble range.
    non_significant_pre2sat = (mean_pre2sat > upper_pre2sat)
    

    return significant_sat2pre, non_significant_pre2sat





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyEDM import CCM

def ccm_significance_test_v3(
    df_sd, 
    df_pre,
    E=4, 
    tau=8, 
    n_ran=20, 
    libSizes="100 200 300 400 500 600 700",
    Tp=0,
    sample=100,
    showPlot=True
):
    """
    Perform a CCM significance test by:
      1) Building a DataFrame with X, Y from df_sd and df_pre.
      2) Running CCM on the real data.
      3) Generating 'n_ran' surrogate versions of X (with random perturbations),
         each time re-running CCM, storing results in ran_ccm_list_xy.
      4) Optionally plotting real vs. surrogate cross mappings.

    Parameters
    ----------
    df_sd : pd.DataFrame
        DataFrame containing at least ["age"] and one data column for X.
    df_pre : pd.DataFrame
        DataFrame containing at least ["age"] and one data column for Y.
    E : int
        Embedding dimension (default=4).
    tau : int
        Time delay (default=8).
    n_ran : int
        Number of surrogate draws (default=20).
    libSizes : str or list
        Library sizes for CCM (default="100 200 300 400 500 600 700").
    Tp : int
        Prediction horizon (default=0).
    sample : int
        Number of bootstrap samples in each CCM call (default=100).
    showPlot : bool
        Whether to show the resulting figure (default=True).

    Returns
    -------
    ccm_out : pd.DataFrame
        CCM output for the real data, containing columns like ["LibSize", "X:Y", "Y:X"].
    ran_ccm_list_xy : list
        List of CCM outputs (DataFrames) from each of the n_ran surrogate runs.
    test_result : any
        Result of the significance histogram test.
    """
    
    def randomize_stadial(stadial_data, seed=None):
        """
        Generate a surrogate time series with the same amplitude (spectrum) as the input stadial_data
        but with randomized phases. This method uses the Fourier transform to preserve the spectral
        structure while removing any specific temporal ordering.
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(stadial_data)
        # Compute the Fourier transform
        fft_data = np.fft.rfft(stadial_data)
        amplitudes = np.abs(fft_data)
        phases = np.angle(fft_data)
        
        # Generate random phases
        random_phases = np.random.uniform(0, 2 * np.pi, len(phases))
        # Preserve the phase of the zero-frequency (DC) component
        random_phases[0] = phases[0]
        # If n is even, preserve the Nyquist component's phase
        if n % 2 == 0:
            random_phases[-1] = phases[-1]
        
        # generate the random amplitudes
        # random_amplitudes = np.random.uniform(0, 1, len(amplitudes))
        # Construct surrogate Fourier coefficients with original amplitudes and randomized phases
        surrogate_fft = amplitudes * np.exp(1j * random_phases)
        # surrogate_fft = random_amplitudes * np.exp(1j * random_phases)
        # Inverse FFT to get the surrogate time series
        surrogate_data = np.fft.irfft(surrogate_fft, n=n)
        
        return surrogate_data

    # Build combined DataFrame: time, X, Y
    # We use the second column in df_sd and df_pre as X and Y, respectively.
    df = pd.DataFrame({
        "Time": df_pre["age"],
        "X":    df_sd[df_sd.columns[1]],
        "Y":    df_pre[df_pre.columns[1]]
    })

    column_name = df_sd.columns[1]
    target_name = df_pre.columns[1]

    # Real-data CCM
    ccm_out = CCM(
        dataFrame   = df,
        E           = E,
        tau         = tau,
        columns     = "X",   # predictor
        target      = "Y",   # target
        libSizes    = libSizes,
        sample      = sample,
        random      = True,
        replacement = False,
        Tp          = Tp
    )

    # Create an array to store the randomly generated time series
    ran_time_series = np.zeros((n_ran, len(df["X"])))
    # Generate surrogate draws
    ran_ccm_list_xy = []
    for i in range(n_ran):
        # Generate surrogate for X with the same spectrum as the original stadial
        X_ran = randomize_stadial(df["X"].values)
        ran_time_series[i] = X_ran

        # Create DataFrame with the same Y but surrogate X
        df_surr = pd.DataFrame({
            "Time": df["Time"],
            "X":    X_ran,
            "Y":    df["Y"].values
        })

        # Run CCM for X->Y on the surrogate data
        out_xy = CCM(
            dataFrame   = df_surr,
            E           = E,
            tau         = tau,
            columns     = "X",
            target      = "Y",
            libSizes    = libSizes,
            sample      = sample,
            random      = True,
            replacement = False,
            Tp          = Tp
        )
        ran_ccm_list_xy.append(out_xy)

    if showPlot:
        # Plot the original time series and the surrogate time series
        fig1, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=100)
        ax.plot(df["Time"], df["X"], label=column_name)
        for i in range(n_ran):
            ax.plot(df["Time"], ran_time_series[i], color='grey', alpha=0.3)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()

    if showPlot:
        # Plot the CCM results for real vs. surrogate data
        fig, ax = plt.subplots(figsize=(4, 4))

        libsize = ran_ccm_list_xy[0]["LibSize"].values

        yx_surrogates = np.column_stack([out_xy["Y:X"].values for out_xy in ran_ccm_list_xy])
        yx_min = np.percentile(yx_surrogates, 5, axis=1)
        yx_max = np.percentile(yx_surrogates, 95, axis=1)

        xy_surrogates = np.column_stack([out_xy["X:Y"].values for out_xy in ran_ccm_list_xy])
        xy_min = np.percentile(xy_surrogates, 5, axis=1)
        xy_max = np.percentile(xy_surrogates, 95, axis=1)

        ax.fill_between(libsize, xy_min, xy_max, color="r", alpha=0.2, label='', edgecolor='none')
        ax.fill_between(libsize, yx_min, yx_max, color="b", alpha=0.2, label='', edgecolor='none')

        ax.plot(ccm_out["LibSize"], ccm_out["Y:X"], "b-",
                label=fr"$\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")
        ax.plot(ccm_out["LibSize"], ccm_out["X:Y"], "r-",
                label=fr"$\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
        
        ax.set_xlim([libsize[0], libsize[-1]])
        ax.set_ylim([-0.15, 1.15])
        ax.set_xlabel("Library Size")
        ax.set_ylabel("Prediction Skill (rho)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    test_result = ccm_significance_hist(ccm_out, ran_ccm_list_xy, column_name=column_name, target_name=target_name, if_plot=showPlot)

    return ccm_out, ran_ccm_list_xy, test_result
