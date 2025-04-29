


import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyEDM import CCM
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

from toolbox import significant_test as st
import importlib
importlib.reload(st)

def global_ccm_significance_map(ds_sat_diff, df_pre, ds_sat_ens_diff='none',
                                   E=5, tau=-1, Tp=0, libSizes="10 20 30 40 50 60 70",
                                   n_ran=10, sample=10, random=False, column_name='sat_diff', function='v2', mode='uni',show_figure=False,
                                   ):
    
    
    nlat = ds_sat_diff.sizes["lat"]
    nlon = ds_sat_diff.sizes["lon"]
    significance_map = np.full((nlat, nlon), False, dtype=bool)

    if function == 'v1':
        for i in range(nlat):
            for j in range(nlon):

                ccm_out, ran_ccm_list, test_result = st.ccm_significance_test_v1(
                    ds_sat_diff, df_pre, ds_sat_ens_diff, lat_idx=i, lon_idx=j,
                    column_name=column_name,
                    E=E, 
                    tau=tau, 
                    n_ran=n_ran, 
                    libSizes=libSizes,
                    Tp=Tp,
                    sample=sample,
                    random=random,
                    flip_pre=False,
                    showPlot=False
                )
                
                # Depending on the mode, set the significance map.
                if mode == 'uni':
                    significance_map[i, j] = test_result[0]
                elif mode == 'bi':
                    significance_map[i, j] = test_result[0] and test_result[1]
                else:
                    raise ValueError("Unknown mode. Choose either 'uni' or 'bi'.")

                print(f"Processed grid point: lat index={i}, lon index={j}, significance={test_result}")

    if function == 'v2':
        for i in range(nlat):
            for j in range(nlon):

                ccm_out, ran_ccm_list, test_result = st.ccm_significance_test_v2(
                    ds_sat_diff, df_pre, lat_idx=i, lon_idx=j,
                    column_name=column_name,
                    E=E, 
                    tau=tau, 
                    n_ran=n_ran, 
                    libSizes=libSizes,
                    Tp=Tp,
                    sample=sample,
                    random=random,
                    flip_pre=False,
                    showPlot=False
                )
                
                # Depending on the mode, set the significance map.
                if mode == 'uni':
                    significance_map[i, j] = test_result[0]
                elif mode == 'bi':
                    significance_map[i, j] = test_result[0] and test_result[1]
                else:
                    raise ValueError("Unknown mode. Choose either 'uni' or 'bi'.")

                print(f"Processed grid point: lat index={i}, lon index={j}, significance={test_result}")

    if function == 'v3':
        for i in range(nlat):
            for j in range(nlon):

                ccm_out, ran_ccm_list, test_result = st.ccm_significance_test_v3(
                    ds_sat_diff, df_pre, lat_idx=i, lon_idx=j,
                    column_name=column_name,
                    E=E, 
                    tau=tau, 
                    n_ran=n_ran, 
                    libSizes=libSizes,
                    Tp=Tp,
                    sample=sample,
                    random=random,
                    flip_pre=False,
                    showPlot=False
                )
                
                # Depending on the mode, set the significance map.
                if mode == 'uni':
                    significance_map[i, j] = test_result[0]
                elif mode == 'bi':
                    significance_map[i, j] = test_result[0] and test_result[1]
                else:
                    raise ValueError("Unknown mode. Choose either 'uni' or 'bi'.")

                print(f"Processed grid point: lat index={i}, lon index={j}, significance={test_result}")

    if show_figure:
        # # Plot the global map.
        # lats = ds_sat_diff["lat"].values
        # lons = ds_sat_diff["lon"].values
        # lon_grid, lat_grid = np.meshgrid(lons, lats)

        # fig = plt.figure(figsize=(12, 6))
        # ax = plt.axes(projection=ccrs.Robinson())
        # ax.coastlines()
        # pcm = ax.pcolormesh(lon_grid, lat_grid, significance_map.astype(int),
        #                       transform=ccrs.PlateCarree(), cmap=plt.cm.Reds, vmin=0, vmax=1)
        # cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05)
        # cb.set_label("Significance (1: True, 0: False)")
        # ax.set_title("Global CCM Significance Map")
        # plt.show()
        plot_sig_map(ds_sat_diff, significance_map, dpi=100)

    return significance_map

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_sig_map(ds_sat_diff, significance_map, dpi=100, camp="Reds", show_grid=False):
    """
    Plots a significance map using a discrete colormap with two colors.
    
    Parameters:
    - ds_sat_diff: xarray.Dataset containing latitude ('lat') and longitude ('lon').
    - significance_map: 2D array of boolean significance values.
    - dpi: resolution of the output figure.
    - camp: colormap to derive the two colors from. Can be a string or a matplotlib colormap object.
            The function uses the left end (camp(0.0)) and the right end (camp(1.0)) of the colormap.
    - show_grid: bool, if True, display a lat-lon grid at 5° intervals with labels.
    """
    # Convert camp to a colormap object if it is provided as a string.
    if isinstance(camp, str):
        camp = plt.get_cmap(camp)
    
    # Extract the left and right endpoint colors from the chosen colormap.
    left_color = camp(0.0)
    right_color = camp(1.0)
    
    # Create a discrete colormap using the two endpoint colors.
    discrete_cmap = ListedColormap([left_color, right_color])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], discrete_cmap.N)
    
    lats = ds_sat_diff["lat"].values
    lons = ds_sat_diff["lon"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=(12, 6), dpi=dpi)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.coastlines()

    # Optionally add gridlines every 5° with latitude/longitude labels.
    if show_grid:
        # Gridlines are plotted in PlateCarree projection.
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 5))
        gl.xlabel_style = {'size': 8, 'color': 'black'}
        gl.ylabel_style = {'size': 8, 'color': 'black'}
        # Disable labels on the top and right sides if desired.
        gl.top_labels = False
        gl.right_labels = False

    pcm = ax.pcolormesh(lon_grid, lat_grid, significance_map.astype(int),
                        transform=ccrs.PlateCarree(), cmap=discrete_cmap, norm=norm)
    
    # Create a colorbar with two ticks for 0 and 1.
    cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5, ticks=[0, 1])
    cb.ax.set_xticklabels(['Non-significant', 'Significant'])
    cb.set_label("p-value < 0.05")
    
    plt.show()


# def plot_sig_map(ds_sat_diff, significance_map, dpi=100, camp="Reds"):
#     """
#     Plots a significance map using a discrete colormap with two colors.
    
#     Parameters:
#     - ds_sat_diff: xarray.Dataset containing latitude ('lat') and longitude ('lon').
#     - significance_map: 2D array of boolean significance values.
#     - dpi: resolution of the output figure.
#     - camp: colormap to derive the two colors from. Can be a string or a matplotlib colormap object.
#             The function uses the left end (camp(0.0)) and the right end (camp(1.0)) of the colormap.
#     """
#     # Convert the camp string to a colormap if necessary.
#     if isinstance(camp, str):
#         camp = plt.get_cmap(camp)
    
#     # Extract the two endpoint colors from the chosen colormap.
#     left_color = camp(0.0)
#     right_color = camp(1.0)
    
#     # Create a discrete colormap with two colors:
#     discrete_cmap = ListedColormap([left_color, right_color])
#     norm = BoundaryNorm([-0.5, 0.5, 1.5], discrete_cmap.N)
    
#     lats = ds_sat_diff["lat"].values
#     lons = ds_sat_diff["lon"].values
#     lon_grid, lat_grid = np.meshgrid(lons, lats)

#     fig = plt.figure(figsize=(12, 6), dpi=dpi)
#     ax = plt.axes(projection=ccrs.Robinson())
#     ax.coastlines()
    
#     pcm = ax.pcolormesh(lon_grid, lat_grid, significance_map.astype(int),
#                         transform=ccrs.PlateCarree(), cmap=discrete_cmap, norm=norm)
    
#     # Create a colorbar with just two ticks representing the two colors.
#     cb = plt.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.5, ticks=[0, 1])
#     cb.ax.set_xticklabels(['Non-significant', 'Significant'])
#     cb.set_label("p-value < 0.05")
    
#     plt.show()




