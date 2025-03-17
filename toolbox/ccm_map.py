import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyEDM import CCM

# Cartopy imports
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def cal_raw_ccm_map(ds_sat, df_pre, E, tau, Tp,v_rage='auto',show_plot=True):
    """
    Compute and plot a global map of CCM skill using pyEDM's CCM.
    
    Parameters:
        E (int): Embedding dimension for CCM.
        tau (int): Time delay.
        Tp (int): Prediction horizon.
    
        ds_sat: xarray Dataset containing "sat", "lat", "lon", and "age".
        df_pre: pandas DataFrame with column "pre".
    """
    
    # Make sure the time series lengths match
    assert len(df_pre) == ds_sat.sizes["age"], "Timeseries length mismatch"

    # ------------------------------------------------
    # Prepare the grid and output array
    # ------------------------------------------------
    nlat = ds_sat.sizes["lat"]
    nlon = ds_sat.sizes["lon"]
    ages = ds_sat["age"].values
    rho_map = np.full((nlat, nlon), np.nan)
    
    # CCM parameter: library sizes (this could also be made an input if desired)
    libSizes = "10 20 30 40 50 60 70 80"
    
    # ------------------------------------------------
    # Loop over all lat/lon grid points and compute CCM skill
    # ------------------------------------------------
    for iLat in range(nlat):
        for iLon in range(nlon):
            # Extract the local "sat" time series
            sat_ts = ds_sat["sat"].isel(lat=iLat, lon=iLon).values

            # Build temporary DataFrame for CCM (predicting "X" using "Y")
            temp_df = pd.DataFrame({
                "Time": ages*-1,
                "X": df_pre["pre"],  # target (what we want to predict)
                "Y": sat_ts,         # predictor
            })

            # Run CCM (Y -> X)
            ccm_out = CCM(
                dataFrame   = temp_df,
                E           = E,
                tau         = tau,
                columns     = "Y",
                target      = "X",
                libSizes    = libSizes,
                sample      = 10,
                random      = True,
                replacement = False,
                Tp          = Tp
            )

            # Extract the mean CCM skill at the largest library size
            largest_L = ccm_out["LibSize"].max()
            mask_last = ccm_out["LibSize"] == largest_L
            rho_at_largest = ccm_out.loc[mask_last, "Y:X"].mean()
            rho_map[iLat, iLon] = rho_at_largest

    # ------------------------------------------------
    # Plot the global map using Cartopy (Robinson projection)
    # ------------------------------------------------
    lats = ds_sat["lat"].values
    lons = ds_sat["lon"].values
    # lats = np.array(ds_sat["lat"].values, dtype=np.float64)
    # lons = np.array(ds_sat["lon"].values, dtype=np.float64)

    if show_plot:
        fig = plt.figure(figsize=(11, 6))
        ax = plt.axes(projection=ccrs.Robinson())
        pcm = ax.pcolormesh(
            lons, lats, rho_map,
            transform=ccrs.PlateCarree(),  # data in lat/lon
            shading="auto"
        )
        ax.coastlines()
        
        if v_rage == 'auto':
            vmin, vmax = np.nanmin(rho_map), np.nanmax(rho_map)
        else:
            vmin, vmax = v_rage
        
        pcm.set_clim(vmin, vmax)
        cb = plt.colorbar(pcm, orientation="horizontal", pad=0.07, shrink=0.8)
        cb.set_label(r"CCM skill $\rho$ (\hat{Pre}$|$M_{sat})")

        
        plt.show()

    return rho_map


import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def postprocess_rho_map(raw_rho_map, significance_map):
    """
    Process the raw rho map by setting non-significant values to NaN.

    Parameters:
    -----------
    raw_rho_map : np.ndarray
        2D array of float values (shape 96 x 144).
    significance_map : np.ndarray
        2D array of boolean values (same shape as raw_rho_map). 
        False (or 0) indicates non-significant data.

    Returns:
    --------
    processed_rho_map : np.ndarray
        A copy of raw_rho_map where all values corresponding to a False 
        entry in significance_map are set to np.nan.
    """
    processed_rho_map = raw_rho_map.copy()
    processed_rho_map[~significance_map] = np.nan
    return processed_rho_map

def plot_rho_map(rho_map, ds_sat, v_range='auto', color_map='viridis'):
    """
    Plot the processed rho map with proper handling of NaN values.

    Parameters:
    -----------
    rho_map : np.ndarray
        2D array representing the processed rho map (may contain NaNs).
    lats : np.ndarray
        1D or 2D array of latitudes corresponding to the rho_map.
    lons : np.ndarray
        1D or 2D array of longitudes corresponding to the rho_map.
    v_range : tuple or str, optional
        A tuple (vmin, vmax) for the color limits. If 'auto', the limits are
        computed from the data using np.nanmin and np.nanmax.
    show_plot : bool, optional
        If True, the plot is displayed immediately.

    Returns:
    --------
    None
    """
    lats = ds_sat["lat"].values
    lons = ds_sat["lon"].values
    fig = plt.figure(figsize=(11, 6))
    ax = plt.axes(projection=ccrs.Robinson())
    pcm = ax.pcolormesh(
        lons, lats, rho_map,
        transform=ccrs.PlateCarree(),  # data are in lat/lon coordinates
        shading="auto",
        cmap=color_map
    )
    ax.coastlines()

    if v_range == 'auto':
        vmin, vmax = np.nanmin(rho_map), np.nanmax(rho_map)
    else:
        vmin, vmax = v_range

    pcm.set_clim(vmin, vmax)
    cb = plt.colorbar(pcm, orientation="horizontal", pad=0.07, shrink=0.6)
    # cb.set_label(r"CCM skill $\rho$ (\hat{Pre}$|$M_{sat})")


