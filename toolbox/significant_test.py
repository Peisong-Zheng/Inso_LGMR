import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyEDM import CCM
from scipy.stats import zscore




def ccm_significance_test_v1(ds_sat, df_pre, ds_sat_ens, column_name='sat_diff', flip_pre=True, lat_idx=80, lon_idx=0,
    E=5, 
    tau=-4, 
    n_ran=20, 
    libSizes="10 20 30 40 50 60 70",
    Tp=0,
    sample=10,
    random =False,
    showPlot=True
):
    
    df_pre=df_pre.copy()
    if flip_pre:
        df_pre['pre']=df_pre['pre'].values*-1

    column_name=column_name
    target_name=df_pre.columns[1]

    time = ds_sat['age'].values
    sat_mean = ds_sat[column_name].isel(lat=lat_idx, lon=lon_idx).values
    sat_ens = ds_sat_ens[column_name].isel(lat=lat_idx, lon=lon_idx).values


    safe_column_name = column_name.replace('_', r'\_')
    safe_target_name = target_name.replace('_', r'\_')
    
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
        plt.figure(figsize=(10, 3))
        for i in range(sat_ens.shape[0]):
            plt.plot(time, zscore(sat_ens[i, :]), color='gray', alpha=0.3)
        plt.plot(time, zscore(sat_mean), color='k', lw=2, label=fr"${{{safe_column_name}}}$")
        if flip_pre:
             plt.plot(df_pre['age'], zscore(df_pre['pre']), color='b', lw=2, label='Pre*-1')
        else:
            plt.plot(df_pre['age'], zscore(df_pre['pre']), color='b', lw=2, label='Pre')
        plt.xlabel("Time (age)")
        plt.ylabel(column_name)
        plt.title(f"lat={int(ds_sat['lat'].values[lat_idx])}, lon={ds_sat['lon'].values[lon_idx]}")
        plt.legend()
        plt.show()

    # half = nTime // 2
    sat_ens_shifted = np.empty_like(sat_ens)
    for i in range(sat_ens.shape[0]):
        ts = sat_ens[i, :]
        # choose a random break point, note there are only N−(E−1)τ data points in the shadow manifold
        # break_point = np.random.randint(len(ts)//5, len(ts)*4//5)
        break_point = np.random.randint(abs(tau)*(E-1), len(ts)-abs(tau)*(E-1))
        randomized_swapped = np.concatenate([ts[break_point:], ts[:break_point]])
        # shifted_ts = np.concatenate((ts[half:], ts[:half]))
        sat_ens_shifted[i, :] = randomized_swapped



    # if showPlot:
    #     plt.figure(figsize=(10, 5))
    #     for i in range(sat_ens_shifted.shape[0]):
    #         plt.plot(time, sat_ens_shifted[i, :], color='orange', alpha=0.2)
    #     plt.plot(time, sat_mean, color='k', lw=2, label=fr"${{{safe_column_name}}}$")
    #     plt.xlabel("Time (age)")
    #     plt.ylabel(column_name)
    #     plt.title("Mean vs. Shifted Ensemble")
    #     plt.legend()
    #     plt.show()
    

    df = pd.DataFrame({
        "Time": df_pre["age"],
        "X":    sat_mean,
        "Y":    df_pre[df_pre.columns[1]]
    })




    # Real-data CCM
    ccm_out = CCM(
        dataFrame   = df,
        E           = E,
        tau         = tau,
        columns     = "X",   # predictor
        target      = "Y",   # target
        libSizes    = libSizes,
        sample      = sample,
        random      = random,
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
                random      = random,
                replacement = False,
                Tp          = Tp
            )
            ran_ccm_list_xy.append(out)
        except Exception as e:
            print(f"Error in ensemble member {i}: {e}")


    if showPlot:
        # create a figure and plot the original time series and the randomized time series
        fig1, ax = plt.subplots(1, 1, figsize=(10, 3),dpi=100)
        ax.plot(df["Time"], df["X"], label=fr"${{{safe_column_name}}}$")
        # ax.plot(df["Time"], df["Y"], label=target_name)
        # plot the randomized time series
        for i in range(n_ran):
            ax.plot(df["Time"], sat_ens_shifted[i,:], color='grey', alpha=0.3)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()



    if showPlot:

        fig, ax = plt.subplots(figsize=(4, 4))

        libsize = ran_ccm_list_xy[0]["LibSize"].values

        # Stack the surrogate data for Y:X and X:Y
        yx_surrogates = np.column_stack([out_xy["Y:X"].values for out_xy in ran_ccm_list_xy])
        # 5th and 95th percentiles for the Y:X surrogates
        yx_min = np.percentile(yx_surrogates, 5, axis=1)
        yx_max = np.percentile(yx_surrogates, 95, axis=1)

        xy_surrogates = np.column_stack([out_xy["X:Y"].values for out_xy in ran_ccm_list_xy])
        # 5th and 95th percentiles for the X:Y surrogates
        xy_min = np.percentile(xy_surrogates, 5, axis=1)
        xy_max = np.percentile(xy_surrogates, 95, axis=1)

        # Fill between for X->Y and Y->X
        ax.fill_between(libsize, xy_min, xy_max, color="r", alpha=0.2, label='', edgecolor='none')
        ax.fill_between(libsize, yx_min, yx_max, color="b", alpha=0.2, label='', edgecolor='none')

        # Use the escaped names in the labels
        ax.plot(ccm_out["LibSize"], ccm_out["Y:X"], "b-",
                label=fr"$\rho$ ($\hat{{{safe_column_name}}}\mid M_{{{safe_target_name}}}$)")

        ax.plot(ccm_out["LibSize"], ccm_out["X:Y"], "r-",
                label=fr"$\rho$ ($\hat{{{safe_target_name}}}\mid M_{{{safe_column_name}}}$)")

        # Set limits and labels
        ax.set_xlim([libsize[0], libsize[-1]])
        ax.set_ylim([-0.15, 1.15])
        ax.set_xlabel("Library Size")
        ax.set_ylabel("Prediction Skill (rho)")
        ax.legend()
        plt.tight_layout()
        plt.show()


    test_result=ccm_significance_hist(ccm_out, ran_ccm_list_xy, column_name=column_name, target_name=target_name, if_plot=showPlot)

    return ccm_out, ran_ccm_list_xy, test_result







def ccm_significance_test_v2(
    ds_sat, df_pre, column_name='sat_diff', flip_pre=True, lat_idx=80, lon_idx=0,
    E=5, 
    tau=-4, 
    n_ran=20, 
    libSizes="10 20 30 40 50 60 70",
    Tp=0,
    sample=10,
    random =False,
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

    df_pre=df_pre.copy()
    if flip_pre:
        df_pre['pre']=df_pre['pre'].values*-1
    
    column_name=column_name
    target_name=df_pre.columns[1]

    time = ds_sat['age'].values
    sat_mean = ds_sat[column_name].isel(lat=lat_idx, lon=lon_idx).values

    def randomize_stadial(stadial_data, seed=None):
        """
        1) Multiply original data by (1 + random variation in [-fraction, fraction]).
        2) Chop in half and rejoin (destroys original time ordering).
        """
        if seed is not None:
            np.random.seed(seed)
        
        # randomly select a break point between 2/10 and 8/10 of the data
        break_point = np.random.randint(abs(tau)*(E-1), len(stadial_data)-abs(tau)*(E-1))
        randomized_swapped = np.concatenate([stadial_data[break_point:], stadial_data[:break_point]])
        
        return randomized_swapped


    df = pd.DataFrame({
        "Time": time,
        "X":    sat_mean,
        "Y":    df_pre[df_pre.columns[1]]
    })



    # Real-data CCM
    ccm_out = CCM(
        dataFrame   = df,
        E           = E,
        tau         = tau,
        columns     = "X",   # predictor
        target      = "Y",   # target
        libSizes    = libSizes,
        sample      = sample,
        random      = random,
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
            random      = random,
            replacement = False,
            Tp          = Tp
        )
        ran_ccm_list_xy.append(out_xy)

    if showPlot:
        # create a figure and plot the original time series and the randomized time series
        fig1, ax = plt.subplots(1, 1, figsize=(10, 3),dpi=100)

        # plot the randomized time series
        for i in range(n_ran):
            ax.plot(df["Time"], zscore(ran_time_series[i]), color='grey', alpha=0.3)
        
        ax.plot(df["Time"], zscore(df["X"]), label=column_name, color='b')
        if flip_pre:
            ax.plot(df["Time"], zscore(df["Y"]), label=target_name+"*-1", color='orange')
        else:
            ax.plot(df["Time"], zscore(df["Y"]), label=target_name, color='orange')
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()

    safe_column_name = column_name.replace('_', r'\_')
    safe_target_name = target_name.replace('_', r'\_')

    if showPlot:

        fig, ax = plt.subplots(figsize=(4, 4))

        libsize = ran_ccm_list_xy[0]["LibSize"].values

        # Stack the surrogate data for Y:X and X:Y
        yx_surrogates = np.column_stack([out_xy["Y:X"].values for out_xy in ran_ccm_list_xy])
        # 5th and 95th percentiles for the Y:X surrogates
        yx_min = np.percentile(yx_surrogates, 5, axis=1)
        yx_max = np.percentile(yx_surrogates, 95, axis=1)

        xy_surrogates = np.column_stack([out_xy["X:Y"].values for out_xy in ran_ccm_list_xy])
        # 5th and 95th percentiles for the X:Y surrogates
        xy_min = np.percentile(xy_surrogates, 5, axis=1)
        xy_max = np.percentile(xy_surrogates, 95, axis=1)

        # Fill between for X->Y and Y->X
        ax.fill_between(libsize, xy_min, xy_max, color="r", alpha=0.2, label='', edgecolor='none')
        ax.fill_between(libsize, yx_min, yx_max, color="b", alpha=0.2, label='', edgecolor='none')

        # Use the escaped names in the labels
        ax.plot(ccm_out["LibSize"], ccm_out["Y:X"], "b-",
                label=fr"$\rho$ ($\hat{{{safe_column_name}}}\mid M_{{{safe_target_name}}}$)")

        ax.plot(ccm_out["LibSize"], ccm_out["X:Y"], "r-",
                label=fr"$\rho$ ($\hat{{{safe_target_name}}}\mid M_{{{safe_column_name}}}$)")

        # Set limits and labels
        ax.set_xlim([libsize[0], libsize[-1]])
        ax.set_ylim([-0.15, 1.15])
        ax.set_xlabel("Library Size")
        ax.set_ylabel("Prediction Skill (rho)")
        ax.legend()
        plt.tight_layout()
        plt.show()



    test_result=ccm_significance_hist(ccm_out, ran_ccm_list_xy, column_name=column_name, target_name=target_name, if_plot=showPlot)

    return ccm_out, ran_ccm_list_xy, test_result





def ccm_significance_test_v3(
    ds_sat, df_pre, column_name='sat_diff', flip_pre=True, lat_idx=80, lon_idx=0,
    E=5, 
    tau=-4, 
    n_ran=20, 
    libSizes="10 20 30 40 50 60 70",
    Tp=0,
    sample=10,
    random =False,
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

    df_pre=df_pre.copy()
    if flip_pre:
        df_pre['pre']=df_pre['pre'].values*-1
    
    column_name=column_name
    target_name=df_pre.columns[1]

    time = ds_sat['age'].values
    sat_mean = ds_sat[column_name].isel(lat=lat_idx, lon=lon_idx).values

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
        
        surrogate_fft = amplitudes * np.exp(1j * random_phases)
        surrogate_data = np.fft.irfft(surrogate_fft, n=n)
        
        return surrogate_data


    df = pd.DataFrame({
        "Time": time,
        "X":    sat_mean,
        "Y":    df_pre[df_pre.columns[1]]
    })



    # Real-data CCM
    ccm_out = CCM(
        dataFrame   = df,
        E           = E,
        tau         = tau,
        columns     = "X",   # predictor
        target      = "Y",   # target
        libSizes    = libSizes,
        sample      = sample,
        random      = random,
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
            random      = random,
            replacement = False,
            Tp          = Tp
        )
        ran_ccm_list_xy.append(out_xy)

    if showPlot:
        # create a figure and plot the original time series and the randomized time series
        fig1, ax = plt.subplots(1, 1, figsize=(10, 3),dpi=100)

        # plot the randomized time series
        for i in range(n_ran):
            ax.plot(df["Time"], zscore(ran_time_series[i]), color='grey', alpha=0.3)
        
        ax.plot(df["Time"], zscore(df["X"]), label=column_name, color='b')
        if flip_pre:
            ax.plot(df["Time"], zscore(df["Y"]), label=target_name+"*-1", color='orange')
        else:
            ax.plot(df["Time"], zscore(df["Y"]), label=target_name, color='orange')
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        plt.show()

    safe_column_name = column_name.replace('_', r'\_')
    safe_target_name = target_name.replace('_', r'\_')

    if showPlot:

        fig, ax = plt.subplots(figsize=(4, 4))

        libsize = ran_ccm_list_xy[0]["LibSize"].values

        # Stack the surrogate data for Y:X and X:Y
        yx_surrogates = np.column_stack([out_xy["Y:X"].values for out_xy in ran_ccm_list_xy])
        # 5th and 95th percentiles for the Y:X surrogates
        yx_min = np.percentile(yx_surrogates, 5, axis=1)
        yx_max = np.percentile(yx_surrogates, 95, axis=1)

        xy_surrogates = np.column_stack([out_xy["X:Y"].values for out_xy in ran_ccm_list_xy])
        # 5th and 95th percentiles for the X:Y surrogates
        xy_min = np.percentile(xy_surrogates, 5, axis=1)
        xy_max = np.percentile(xy_surrogates, 95, axis=1)

        # Fill between for X->Y and Y->X
        ax.fill_between(libsize, xy_min, xy_max, color="r", alpha=0.2, label='', edgecolor='none')
        ax.fill_between(libsize, yx_min, yx_max, color="b", alpha=0.2, label='', edgecolor='none')

        # Use the escaped names in the labels
        ax.plot(ccm_out["LibSize"], ccm_out["Y:X"], "b-",
                label=fr"$\rho$ ($\hat{{{safe_column_name}}}\mid M_{{{safe_target_name}}}$)")

        ax.plot(ccm_out["LibSize"], ccm_out["X:Y"], "r-",
                label=fr"$\rho$ ($\hat{{{safe_target_name}}}\mid M_{{{safe_column_name}}}$)")

        # Set limits and labels
        ax.set_xlim([libsize[0], libsize[-1]])
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





