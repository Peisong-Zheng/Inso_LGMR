import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def center_data(ds):
    if 'sat' not in ds:
        raise ValueError("The input xarray dataset does not contain a 'sat' variable.")
    
    # Calculate the mean across the 'age' dimension
    mean_data = ds['sat'].mean(dim='age')
    
    # Subtract the mean from the data
    centered_data = ds['sat'] - mean_data
    
    # Add the centered data back to the dataset
    ds['sat_centered'] = centered_data
    return ds



def apply_weighting(ds):
    # Check for necessary variables
    for var in ['lat', 'sat_centered']:
        if var not in ds:
            raise ValueError(f"The input xarray dataset does not contain a '{var}' variable.")
            
    # Calculate the weight
    weight = np.cos(np.deg2rad(ds['lat']))
    print('shape of the weight:', weight.shape)

    # Set weight to 0 if it is smaller than 0
    weight = xr.where(weight < 0, 0, weight)

    # Add the weight to the dataset
    ds['weight'] = weight
    
    # Multiply 'sat_centered' with the weight
    sat_anomalies_weighted = ds['sat_centered'] * weight
    
    # Add 'sat_anomalies_weighted' as a new data variable in the dataset
    ds['sat_centered_weighted'] = sat_anomalies_weighted
    
    # Extract values and reshape
    sat_centered_weighted = ds['sat_centered_weighted'].values
    sat_centered_weighted = sat_centered_weighted.reshape(len(ds['age']), -1)
    print('shape of the reshaped sat_centered_weighted:', sat_centered_weighted.shape)
    
    return ds



def plot_sat_variables(ds, lat_idx, lon_idx):
    if not all(var in ds for var in ['sat', 'sat_centered', 'sat_centered_weighted']):
        raise ValueError("The dataset does not contain all required variables ('sat', 'sat_centered', 'sat_centered_weighted')")
    
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(ds['age'], ds['sat'][:, lat_idx, lon_idx], label='sat')
    ax.plot(ds['age'], ds['sat_centered'][:, lat_idx, lon_idx], label='sat_centered')
    ax.plot(ds['age'], ds['sat_centered_weighted'][:, lat_idx, lon_idx], label='sat_centered_weighted')

    # plot a line at y=0
    ax.axhline(y=0, color='black', linestyle='--')

    ax.set_xlabel('age')
    ax.set_ylabel('sat')
    ax.legend()

    plt.show()



def sat_PCA(ds):
    if 'sat_centered_weighted' not in ds:
        raise ValueError("The dataset does not contain 'sat_centered_weighted' variable.")
    
    if 'age' not in ds.dims:
        raise ValueError("The dataset does not contain 'age' dimension.")

    # Reshape the sat_centered_weighted data
    sat_centered_weighted = ds['sat_centered_weighted'].values
    original_shape = sat_centered_weighted.shape
    sat_centered_weighted = sat_centered_weighted.reshape(original_shape[0], -1).T
    print('shape of the reshaped sat_centered_weighted:', sat_centered_weighted.shape)

    # Perform Singular Value Decomposition
    u, s, vh = np.linalg.svd(sat_centered_weighted, full_matrices=True)
    print('shape of u, s, vh:', u.shape, s.shape, vh.shape)
    
    # Extract the first two EOFs
    eofs = u[:, :2]
    print('shape of EOFs:', eofs.shape)
    
    # Calculate the Principal Components
    pcs = sat_centered_weighted.T.dot(eofs)
    print('shape of PCs:', pcs.shape)
    
    # Calculate the variance explained by the first 10 PCs
    exp_variance = s**2 / np.sum(s**2)
    
    return exp_variance, eofs, pcs





import numpy as np
from sklearn.decomposition import PCA

def sat_PCA_sklearn(ds):
    if 'sat_centered_weighted' not in ds:
        raise ValueError("The dataset does not contain 'sat_centered_weighted' variable.")
    
    if 'age' not in ds.dims:
        raise ValueError("The dataset does not contain 'age' dimension.")

    # Reshape the sat_centered_weighted data
    sat_centered_weighted = ds['sat_centered_weighted'].values
    original_shape = sat_centered_weighted.shape
    sat_centered_weighted_reshaped = sat_centered_weighted.reshape(original_shape[0], -1)
    print('shape of the reshaped sat_centered_weighted:', sat_centered_weighted_reshaped.shape)

    # Perform PCA
    pca = PCA(n_components=2)  # specify the number of components
    pcs = pca.fit_transform(sat_centered_weighted_reshaped)
    print('shape of PCs:', pcs.shape)

    # Calculate Explained Variance
    exp_variance = pca.explained_variance_ratio_
    print('Explained variance:', exp_variance)
    
    # Calculate the EOFs
    eofs = pca.components_.T
    print('shape of EOFs:', eofs.shape)
    
    # # Reshape EOFs back to original grid shape
    # eofs = eofs.reshape((2,) + original_shape[1:])
    # print('shape of reshaped EOFs:', eofs.shape)

    return exp_variance, eofs, pcs




# def plot_pcs(age, pcs, variance_explained):
#     if pcs.shape[1] < 2:
#         raise ValueError("The input 'pcs' must have at least 2 columns (principal components).")
    
#     if len(variance_explained) < 2:
#         raise ValueError("The input 'variance_explained' must have at least 2 values.")

#     fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

#     ax.plot(age, -1 * pcs[:, 0], label='PC1, v_exp={:.2f}'.format(variance_explained[0]))
#     ax.plot(age, pcs[:, 1], label='PC2, v_exp={:.2f}'.format(variance_explained[1]))

#     # reverse the x-axis
#     ax.invert_xaxis()

#     ax.set_xlabel('age')
#     ax.set_ylabel('PCs')
#     ax.legend()
#     plt.show()


def plot_pcs(age, pcs, variance_explained):
    if pcs.shape[1] < 2:
        raise ValueError("The input 'pcs' must have at least 2 columns (principal components).")
    
    if len(variance_explained) < 2:
        raise ValueError("The input 'variance_explained' must have at least 2 values.")

    fig, ax = plt.subplots(figsize=(4.5, 3), dpi=300)

    ax.plot(age, -1 * pcs[:, 0], label='PC1, v_exp={:.2f}'.format(variance_explained[0]))
    ax.plot(age, pcs[:, 1], label='PC2, v_exp={:.2f}'.format(variance_explained[1]))

    # reverse the x-axis
    ax.invert_xaxis()
    # set the linewidth of the box
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    ax.set_xlabel('Age (yr BP)')
    ax.set_ylabel('PCs')
    ax.legend()
    plt.show()
    return fig, ax



def plot_eof_scatter(eofs):
    if eofs.shape[1] < 2:
        raise ValueError("The input 'eofs' must have at least 2 columns.")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(eofs[:, 0], eofs[:, 1])
    
    ax.set_xlabel('EOF1')
    ax.set_ylabel('EOF2')
    ax.set_title('Scatter plot of EOF1 vs EOF2')
    
    plt.show()




def plot_eof_map(eofs, lat, lon):
    if eofs.shape[1] < 2:
        raise ValueError("The input 'eofs' must have at least 2 columns (EOFs).")
    
    if len(eofs) != len(lat) * len(lon):
        raise ValueError("The length of 'eofs' must be equal to len(lat) * len(lon).")

    # Reshape EOF1 and EOF2 to 2D array
    eof1 = eofs[:, 0].reshape(len(lat), len(lon))
    eof2 = eofs[:, 1].reshape(len(lat), len(lon))

    # create a figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(8, 9), subplot_kw=dict(projection=ccrs.Robinson()), dpi=600)

    for i, eof in enumerate([eof1, eof2]):
        ax = axs[i]
        # add coastline and gridlines
        ax.add_feature(cfeature.COASTLINE)

        # Configure the gridlines
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Rotate longitude labels
        gl.xlabel_style = {'rotation': 90}

        # plot heatmap
        cmap = plt.cm.get_cmap('coolwarm')
        im = ax.pcolormesh(lon, lat, eof, transform=ccrs.PlateCarree(), cmap=cmap, vmin=-0.04, vmax=0.04, shading='auto')
        ax.set_title(f'EOF{i+1}')

        # add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.5)
        cbar.ax.set_ylabel('Loading')

    # adjust the space between subplots
    fig.subplots_adjust(hspace=0.005)

    plt.show()
    return fig, axs

# def plot_eof_map(eofs, lat, lon):
#     if eofs.shape[1] < 2:
#         raise ValueError("The input 'eofs' must have at least 2 columns (EOFs).")
    
#     if len(eofs) != len(lat) * len(lon):
#         raise ValueError("The length of 'eofs' must be equal to len(lat) * len(lon).")

#     # Reshape EOF1 and EOF2 to 2D array
#     eof1 = eofs[:, 0].reshape(len(lat), len(lon))*-1
#     eof2 = eofs[:, 1].reshape(len(lat), len(lon))

#     # create a figure and axes
#     fig, axs = plt.subplots(2, 1, figsize=(8, 9), subplot_kw=dict(projection=ccrs.Robinson()), dpi=300)

#     for i, eof in enumerate([eof1, eof2]):
#         ax = axs[i]
#         # add coastline and gridlines
#         ax.add_feature(cfeature.COASTLINE)
#         ax.gridlines(draw_labels=False)

#         # plot heatmap
#         cmap = plt.cm.get_cmap('coolwarm')
#         im = ax.pcolormesh(lon, lat, eof, transform=ccrs.PlateCarree(), cmap=cmap, vmin=-0.04, vmax=0.04, shading='auto')
#         ax.set_title(f'EOF{i+1}')

#         # add colorbar
#         cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.5)
#         cbar.ax.set_ylabel('Loading')

#     # adjust the space between subplots
#     fig.subplots_adjust(hspace=0.05)

#     plt.show()

