import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as patheffects
from matplotlib import gridspec

import astropy.units as u

from astropy_healpix import HEALPix
from astropy.wcs import WCS
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from reproject import reproject_from_healpix
from astropy.coordinates import SkyCoord

from scipy.stats import binned_statistic_2d
from matplotlib import colors

from pylab import cm
import math


def plot_2dfunc(
    f, lims=[(0, 1), (0, 1)],
    xlabel="$x$",
    ylabel="$y$",
    title=None,
    cmap='viridis',
    fig=None, axs=None,
    **kwargs
):
    """
    Plots a 2d function f(x, y) with the given limits.

    Inputs:
        f (callable): The function to plot
        lims (list): A list of tuples specifying the limits of the plot
        **kwargs: Additional key words to be passed to imshow. Usually used for
            finetuning color maps
    """
    if fig is None:
        fig, axs = plt.subplots(
            1, 2,
            figsize=(3.15, 3),
            gridspec_kw=dict(width_ratios=[3, 0.15])
        )
    ax, cax = axs

    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    grid_size = 256
    x = np.linspace(xmin, xmax, grid_size + 1)
    y = np.linspace(ymin, ymax, grid_size + 1)
    X, Y = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]))

    z = np.reshape(f(X.ravel(), Y.ravel()), X.shape)

    im = ax.imshow(
        z,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        **kwargs
    )
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return fig, axs


def plot_2dhist(
    x, y, values=None, operation="count",
    bins=(64, 64),
    xlabel="$x$",
    ylabel="$y$",
    lims=None,
    title=None,
    cmap='viridis',
    fig=None, axs=None,
    **kwargs
):
    """
    Makes a 2d histogram of set of points with coordinates x, y. Optionally the histogram
    can be replaced with a custom operation for reducing the values in the bins.
    The limits of both x, y, and the colorscale are inferred from the data.

    Inputs:
        x, y (np.array): Arrays showing the x-, and y-coordinates of the data to be plotted
        values (np.array): The values of the data. Used when operation is not 'count'
        **kwargs: Additional key words to be passed to imshow. Usually used for
            finetuning color maps
    """
    if fig is None:
        fig, axs = plt.subplots(
            2, 1,
            figsize=(4, 4.2),
            gridspec_kw=dict(width_ratios=[3], height_ratios=[0.2, 4]),
            layout='constrained'
        )
    cax, ax = axs

    # Get the plot limits
    if lims is None:
        lims = []
        k = 1.1
        for x_ in [x, y]:
            xlim = np.percentile(x_, [1.0, 99.0])
            w = xlim[1] - xlim[0]
            xlim = [xlim[0] - 0.2 * w, xlim[1] + 0.2 * w]
            lims.append(xlim)
    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    x_bins = np.linspace(xmin, xmax, bins[0])
    y_bins = np.linspace(ymin, ymax, bins[1])

    ret = binned_statistic_2d(
        x, y, values, statistic=operation, bins=[x_bins, y_bins]
    )

    # Choose a suitable colormap
    if operation == "count":
        kwargs["vmin"] = 0

    im = ax.imshow(
        ret.statistic.T,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        aspect="auto",
        cmap=cmap,
        **kwargs,
    )
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.locator_params(nbins=5)
    if title is not None:
        cax.set_title(title)

    return fig, axs


def mollweide_wcs(w, frame):
    """ A utility function that returns a WCS object for a Mollweide projection
    with the given width and frame. The height of the projection is width/2.
    """
    coordsys, ctype0, ctype1 = {
        "galactic": ("GAL", "GLON-MOL", "GLAT-MOL"),
        "icrs": ("EQU", "RA---MOL", "DEC--MOL"),
    }[frame]
    target_header = dict(
        naxis=2,
        naxis1=w,
        naxis2=w // 2,
        ctype1=ctype0,
        crpix1=w // 2 + 0.5,
        crval1=0.0,
        cdelt1=-0.675 * 480 / w,
        cunit1="deg",
        ctype2=ctype1,
        crpix2=w // 4 + 0.5,
        crval2=0.0,
        cdelt2=0.675 * 480 / w,
        cunit2="deg",
        coordsys=coordsys,
    )
    wcs = WCS(target_header)
    return wcs


def plot_mollweide(
    healpix_data=None,
    fig=None,
    w=1024,
    input_frame="galactic",
    plot_frame="galactic",
    imshow_kw=dict(cmap="viridis"),
):
    """
    Makes a Mollweide projection plot of the given healpix data. This function
    is usually called by other plot_healpix_* functions, as calculating the
    healpix data can be a bit cumbersome.

    Inputs:
        healpix_data (np.array): The data to be plotted. If equal to None, an empty
            mollweide axis is returned
        fig (matplotlib.figure.Figure): The figure to plot on. If equal to None, a new
            figure is created
        w (int): The width of the plot in pixels
        input_frame (str): The frame of the healpix data. Can be either 'galactic' or 'icrs'
        plot_frame (str): The frame of the plot. Can be either 'galactic' or 'icrs'
        cax_kw (dict): Additional key words to be passed to the colorbar
    """
    # Initiate the figure and the axis
    wcs = mollweide_wcs(w, plot_frame)
    if healpix_data is None:
        # Just create an empty axis and return
        if fig is None:
            fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 1, width_ratios=[8])
        ax = fig.add_subplot(
            gs[0],
            projection=wcs,
            frame_class=EllipticalFrame,
        )
        ax.set_xlim(0, w)
        ax.set_ylim(0, w // 2)
    else:
        if fig is None:
            fig = plt.figure(figsize=(8.15, 4), layout="tight")
        gs = gridspec.GridSpec(1, 2, width_ratios=[8, 0.15])
        ax = fig.add_subplot(
            gs[0],
            projection=wcs,
            frame_class=EllipticalFrame,
        )
        cax = fig.add_subplot(gs[1])

    # Makes characters have a black outline (with some small white outline on top)
    pe = [
        patheffects.withStroke(linewidth=1.2, foreground="black"),
        patheffects.withStroke(linewidth=0.2, foreground="white"),
    ]

    coord0, coord1 = {"galactic": ("glon", "glat"), "icrs": ("ra", "dec")}[plot_frame]
    ax.coords[coord0].set_ticklabel(color="white", path_effects=pe, fontsize=10)
    ax.coords[coord1].set_ticklabel(fontsize=10)
    ax.coords[coord0].set_ticks_visible(False)
    ax.coords[coord1].set_ticks_visible(False)
    ax.coords.grid(color="gray", alpha=0.2)
    if healpix_data is None:
        return fig, ax

    cax.yaxis.set_label_position("right")

    array, footprint = reproject_from_healpix(
        (healpix_data, input_frame),
        wcs,
        nested=False,
        shape_out=(w // 2, w),
        order="nearest-neighbor",
    )

    im = ax.imshow(array, **imshow_kw)
    fig.colorbar(im, cax=cax, orientation="vertical")

    return fig, (ax, cax)


def plot_mollweide_hist(
    l_deg, b_deg,
    values=None, operation=np.mean,
    fig=None,
    w=1024,
    input_frame="galactic",
    plot_frame="galactic",
    nside=32,
    title=None,
    imshow_kw=dict(cmap="viridis"),
):
    """
    Makes an aggregated skyplot of values based on the given operation.
    values are binned per healpixel and the operation is applied to the values
    in each bin.

    Inputs:
        l_deg, b_deg (np.ndarray): The longitude and latitude of the datapoints
            in degrees.
        y_plot (np.array): Value to aggregate. If equal to None, the sum
            of datapoints per bin is returned instead
        operation (callable): The operation to apply on the values of y_plot in a healpixel bin
        w (int): The width of the plot in pixels
        input_frame (str): The frame of the healpix data. Can be either 'galactic' or 'icrs'
        plot_frame (str): The frame of the plot. Can be either 'galactic' or 'icrs'
        nside (int): The nside of the healpix grid
        title (str): The title of the plot
        imshow_kw (dict): Additional key words to be passed to imshow
    """
    def np_group_and_reduce(values_to_groupby, y, n, operation=np.mean):
        # Groups the values of y based on values_to_groupby and returns
        #     the reduced group values based on the operation.
        # Currently assumes that values_to_groupby has values between [0, n)
        sort_idx = values_to_groupby.argsort()
        values_to_groupby, y = values_to_groupby[sort_idx], y[sort_idx]

        u1, u2 = np.unique(values_to_groupby, return_index=True)
        group_ys = np.split(y, u2[1:])

        group_results = np.zeros(n)
        for i in range(len(u1)):
            group_results[u1[i]] = operation(group_ys[i])

        return group_results

    hp = HEALPix(nside=nside, frame=input_frame)

    ihealpix = hp.lonlat_to_healpix(l_deg * u.deg, b_deg * u.deg)
    if values is None:
        values = np.ones(len(l_deg))
        operation = "count"
    if operation == "count":
        operation = np.sum
    hp_values = np_group_and_reduce(
        ihealpix, values, hp.npix, operation
    )  # binned and reduced values of y per healpixel

    fig, axs = plot_mollweide(
        hp_values, w=w, input_frame=input_frame, plot_frame=plot_frame,
        fig=fig, imshow_kw=imshow_kw
    )

    if title is not None:
        axs[0].set_title(title)

    return fig, axs


def plot_mollweide_func(
    f,
    fig=None,
    w=1024,
    input_frame="galactic",
    plot_frame="galactic",
    nside=32,
    title=None,
    imshow_kw=dict(cmap="viridis"),
):
    """
    Makes a skyplot of the given function f. The function is evaluated on a
    healpix grid and the result is plotted using imshow.

    Inputs:
        f (callable): The function to plot
        w (int): The width of the plot in pixels
        input_frame (str): The frame of the healpix data. Can be either 'galactic' or 'icrs'
        plot_frame (str): The frame of the plot. Can be either 'galactic' or 'icrs'
        nside (int): The nside of the healpix grid
        title (str): The title of the plot
        imshow_kw (dict): Additional key words to be passed to imshow
    """
    hp = HEALPix(nside=nside, frame=input_frame)
    l, b = hp.healpix_to_lonlat(np.arange(hp.npix))
    hp_values = f(l.deg, b.deg)

    fig, axs = plot_mollweide(
        hp_values, w=w, input_frame=input_frame, plot_frame=plot_frame,
        fig=fig, imshow_kw=imshow_kw
    )

    if title is not None:
        axs[0].set_title(title)

    return fig, axs


def plot_mollweide_scatter(
    *args,
    fig=None,
    ax=None,
    w=1024,
    input_frame="galactic",
    plot_frame="galactic",
    title=None,
    **kwargs,
):
    """
    Makes a scatter plot in mollweide projection. The coordinates of the points
    are to be in degrees.

    Inputs:
        *args: The x, y coordinates of the points to be plotted, in degrees
        w (int): The width of the plot in pixels
        input_frame (str): The frame of the scatter data. Can be either 'galactic' or 'icrs'
        plot_frame (str): The frame of the plot. Can be either 'galactic' or 'icrs'
        **kwargs: Additional key words to be passed to scatter
    """
    if fig is None:
        fig, ax = plot_mollweide(w=w, input_frame=input_frame, plot_frame=plot_frame)
    ax.scatter(*args, **kwargs, transform=ax.get_transform("world"))
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_residual_y_ypred(
    y,
    y_pred,
    lims=None,
    bins=(32, 32),
    xlabel="$y$",
    ylabel="$y - y_\mathrm{pred}$",
    title=None,
    slopes=[0, 0.2, 0.5, 1.0],
    fig=None, ax=None
):
    """
    Makes a residual scatter plot comparing y with the deviation from the prediction y_pred.
    Also plots straight lines corresponding to different fractions of deviations from y

    Inputs:
        y (np.array): An array of values
        y_pred (np.array): The model prediction for y
        slopes (list or np.array): The slopes for which deviation lines are marked.
            These are marked at the right hand side of the plot
    """
    import labellines

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    y_res = y_pred - y

    if lims is None:
        lims = []
        k = 0.3
        for vals in [y, y_res]:
            xlim = np.percentile(vals, [1.0, 99.0])
            w = xlim[1] - xlim[0]
            xlim = [xlim[0] - k * w, xlim[1] + k * w]
            lims.append(xlim)
    xlim, ylim = lims
    xmin, xmax = xlim
    ymin, ymax = ylim

    bins_x = np.linspace(*xlim, bins[0])
    bins_y = np.linspace(*ylim, bins[1])

    n, x_edges, _ = np.histogram2d(y, y_res, bins=(bins_x, bins_y))
    norm = np.max(n, axis=1) + 1e-10
    n /= norm[:, None]
    ax.imshow(
        n.T,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=xlim + ylim,
        cmap=plt.cm.Greys,
    )

    for slope in slopes:
        LIM = 10000
        ax.plot(
            [-LIM, LIM],
            [-LIM * slope, LIM * slope],
            color="grey",
            ls="--",
            lw=0.5,
            label=f"{100*slope:.1f}\\%",
        )
        if slope != 0:
            ax.plot(
                [-LIM, LIM],
                [LIM * slope, -LIM * slope],
                color="grey",
                ls="--",
                lw=0.5,
                label=f"{-100*slope:.1f}\\%",
            )

    # put the labels at 90% along x. TODO: can be improved so that they're also placed on y
    labellines.labelLines(
        ax.get_lines(),
        fontsize=8,
        xvals=[xmax - 0.1 * (xmax - xmin) for _ in range(len(ax.get_lines()))],
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    return fig, ax
