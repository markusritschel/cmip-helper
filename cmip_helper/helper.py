# !/usr/bin/env python
# -*- coding utf-8 -*-
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  kontakt@markusritschel.de
# Date:   26/08/2020
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import xarray as xr
import numpy as np


# https://nordicesmhub.github.io/NEGI-Abisko-2019/training/Example_model_global_arctic_average.html
def weighted_average(xa: xr.DataArray,
                     dim=None,
                     weights: xr.DataArray = None,
                     mask: xr.DataArray = None):
    """
    This function will average
    :param xa: dataArray
    :param dim: dimension or list of dimensions. e.g. 'lat' or ['lat','lon','time']
    :param weights: weights (as xarray)
    :param mask: mask (as xarray), True where values to be masked.
    :return: masked average xarray
    """
    # lets make a copy of the xa
    xa_copy: xr.DataArray = xa.copy()

    if mask is not None:
        xa_weighted_average = __weighted_average_with_mask(
            dim, mask, weights, xa, xa_copy
            )
    elif weights is not None:
        xa_weighted_average = __weighted_average(
            dim, weights, xa, xa_copy
            )
    else:
        xa_weighted_average = xa.mean(dim)

    return xa_weighted_average


def __weighted_average(dim, weights, xa, xa_copy):
    """helper function for masked_average"""
    _, weights_all_dims = xr.broadcast(xa, weights)  # broadcast to all dims
    x_times_w = xa_copy*weights_all_dims
    xw_sum = x_times_w.sum(dim)
    x_tot = weights_all_dims.where(xa_copy.notnull()).sum(dim=dim)
    xa_weighted_average = xw_sum/x_tot

    return xa_weighted_average


def __weighted_average_with_mask(dim, mask, weights, xa, xa_copy):
    """helper function for masked_average"""
    _, mask_all_dims = xr.broadcast(xa, mask)  # broadcast to all dims
    xa_copy = xa_copy.where(np.logical_not(mask))
    if weights is not None:
        _, weights_all_dims = xr.broadcast(xa, weights)  # broadcast to all dims
        weights_all_dims = weights_all_dims.where(~mask_all_dims)
        x_times_w = xa_copy*weights_all_dims
        xw_sum = x_times_w.sum(dim=dim)
        x_tot = weights_all_dims.where(xa_copy.notnull()).sum(dim=dim)
        xa_weighted_average = xw_sum/x_tot
    else:
        xa_weighted_average = xa_copy.mean(dim)

    return xa_weighted_average


#######################
# https://github.com/pydata/xarray/issues/422
def average(data, dim=None, weights=None):
    """
    weighted average for xray objects

    Parameters
    ----------
    data : Dataset or DataArray
        the xray object to average over
    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset or DataArray
        New xray object with average applied to its data and the indicated
        dimension(s) removed.

    """

    if isinstance(data, xray.Dataset):
        return average_ds(data, dim, weights)
    elif isinstance(data, xray.DataArray):
        return average_da(data, dim, weights)
    else:
        raise ValueError("date must be an xarray Dataset or DataArray")
