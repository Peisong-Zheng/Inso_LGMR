
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def find_optimal_Tp(df_sd, df_pre,
                    E=4, tau=-1,
                    libSizes="10 20 30 40 50 60 70",
                    sample=20,
                    Tps=range(-10, 11),
                    interactive=False):
    """
    Calls ccm_DOXmapForcing for each Tp in [Tps], collects the maximum rho 
    for forward (X->Y) and reverse (Y->X) directions across all library sizes, 
    and plots these maxima vs. Tp.

    Parameters
    ----------
    df_sd : pd.DataFrame
        DataFrame containing at least ['age'] plus one other column (predictor).
    df_pre : pd.DataFrame
        DataFrame containing at least ['age'] plus one other column (target).
    E : int
        Embedding dimension (default=4).
    tau : int
        Time delay (default=8).
    libSizes : str or list
        Library sizes for CCM (default="100 200 300 400 500 600 700").
    sample : int
        Number of bootstrap samples for CCM calls (default=20).
    Tps : range or list
        Range/list of Tp values to test (default=range(-10, 11)).
    interactive : bool
        If True, plot an interactive Plotly figure instead of Matplotlib.

    Returns
    -------
    Tps_list : list of int
        The range of Tp values tested.
    forward_max_rhos : list of float
        The maximum rho (X->Y) for each Tp.
    reverse_max_rhos : list of float
        The maximum rho (Y->X) for each Tp.
    """

    column_name = df_sd.columns[1]
    target_name = df_pre.columns[1]
    forward_key = f"{column_name}:{target_name}"
    reverse_key = f"{target_name}:{column_name}"

    forward_max_rhos = []
    reverse_max_rhos = []

    # Loop over each Tp, calling the ccm_DOXmapForcing function once.
    for Tp in Tps:
        # Call ccm_DOXmapForcing with showPlot=False 
        ccm_out = ccm_DOXmapForcing(
            df_sd      = df_sd,
            df_pre     = df_pre,
            E          = E,
            tau        = tau,
            libSizes   = libSizes,
            Tp         = Tp,
            sample     = sample,
            showPlot   = False
        )

        # Extract max cross mapping skill for X->Y and Y->X
        forward_rho = ccm_out[forward_key].max()  # X->Y
        reverse_rho = ccm_out[reverse_key].max()  # Y->X

        forward_max_rhos.append(forward_rho)
        reverse_max_rhos.append(reverse_rho)

    # Convert Tps to a list for final return
    Tps_list = list(Tps)

    # ----- PLOTTING -----
    if interactive:
        # Use Plotly for an interactive figure
        fig = go.Figure()

        # Forward: X->Y
        fig.add_trace(go.Scatter(
            x=Tps_list, 
            y=forward_max_rhos, 
            mode='lines+markers',
            name=fr"CCM skill ρ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)"
        ))
        # Reverse: Y->X
        fig.add_trace(go.Scatter(
            x=Tps_list, 
            y=reverse_max_rhos, 
            mode='lines+markers',
            name=fr"CCM skill ρ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)"
        ))

        fig.update_layout(
            title="Max CCM skill vs. Tp (Interactive)",
            xaxis_title="Tp",
            yaxis_title="Max rho (across libSizes)",
            width=900,
            height=400
        )
        # Set discrete ticks so that all Tps appear
        fig.update_xaxes(tickmode='array', tickvals=Tps_list)

        fig.show()

    else:
        # Use Matplotlib for a static figure
        plt.figure(figsize=(7,5))
        plt.plot(Tps_list, forward_max_rhos, "ro-",
                 label=fr"CCM skill $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
        plt.plot(Tps_list, reverse_max_rhos, "bo-",
                 label=fr"CCM skill $\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")

        plt.xticks(Tps_list)  # set xticks to all Tps
        plt.xlabel("Tp")
        plt.ylabel("Max rho (across libSizes)")
        plt.title("Max CCM skill vs. Tp")
        plt.legend()
        plt.tight_layout()
        plt.show()
    # ----- END PLOTTING -----

    return Tps_list, forward_max_rhos, reverse_max_rhos



import pandas as pd
import numpy as np
from pyEDM import CCM
import matplotlib.pyplot as plt
from scipy.stats import zscore

def ccm_DOXmapForcing(df_sd, df_pre,
                      E=4, tau=8,
                      libSizes="100 200 300 400 500 600 700",
                      Tp=0,
                      sample=20,
                      random = True,
                      showPlot=True):
    """
    Perform Convergent Cross Mapping (CCM) between the second columns of
    df_sd (predictor) and df_pre (target). Plots the time series and the
    CCM skill curves if showPlot is True.

    Parameters
    ----------
    df_sd : pd.DataFrame
        DataFrame containing at least ['age'] plus one other column of interest.
    df_pre : pd.DataFrame
        DataFrame containing at least ['age'] plus one other column of interest.
    E : int
        Embedding dimension (default=4).
    tau : int
        Time delay for embedding (default=8).
    libSizes : str or list
        Library sizes to use in CCM (default="100 200 300 400 500 600 700").
    sample : int
        Number of bootstrap samples (default=20).
    showPlot : bool
        Whether to show the plots (time series & CCM skill). Default=True.

    Returns
    -------
    ccm_out : pd.DataFrame (or CCM object depending on your pyEDM version)
        The result of the CCM call. Typically a DataFrame with columns:
        ['LibSize', 'X:Y', 'Y:X'] if your pyEDM version returns that directly.
    """

    # Identify the column names to use for predictor and target
    # (the second column in each DataFrame)
    ages         = df_pre["age"]
    column_name  = df_sd.columns[1]
    target_name  = df_pre.columns[1]

    # Combine into a single DataFrame for CCM
    df = pd.DataFrame({
        "Time":        ages,
        column_name:   df_sd[column_name],  # predictor
        target_name:   df_pre[target_name], # target
    })

    # Optional: plot the time series
    if showPlot:
        fig, ax = plt.subplots()
        ax.plot(df["Time"], zscore(df[column_name]), "r-", label=column_name)
        ax.plot(df["Time"], zscore(df[target_name]), "b-", label=target_name)
        ax.set_xlabel("Time (age)")
        ax.set_ylabel("Normalized values")
        ax.legend()
        plt.title("Time Series of Predictor vs. Target")
        plt.show()

    # Perform the CCM call
    ccm_out = CCM(
        dataFrame   = df,
        E           = E,
        tau         = tau,
        columns     = column_name,   # predictor
        target      = target_name,   # target
        libSizes    = libSizes,
        sample      = sample,
        random      = random,
        replacement = False,
        Tp          = Tp
    )

    # Optional: plot the cross mapping skill
    if showPlot:
        # Construct keys for the forward/reverse cross mapping
        forward_key = f"{column_name}:{target_name}"
        reverse_key = f"{target_name}:{column_name}"

        fig, ax = plt.subplots()
        ax.plot(ccm_out["LibSize"], ccm_out[forward_key],
                "ro-", label=fr"CCM skill $\rho$ ($\hat{{{target_name}}}\mid M_{{{column_name}}}$)")
        ax.plot(ccm_out["LibSize"], ccm_out[reverse_key],
                "bo-", label=fr"CCM skill $\rho$ ($\hat{{{column_name}}}\mid M_{{{target_name}}}$)")
        ax.set_xlabel("Library Size")
        ax.set_ylabel("Prediction skill (rho)")
        ax.legend()
        plt.title("CCM Skill Curves")
        plt.show()

    return ccm_out
