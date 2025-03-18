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

# Example usage:
# result = ccm_significance_test(ccm_out, ensemble_ccm)
# print("Significant (SAT significantly predicts pre, while pre does not predict SAT):", result)
