# !/usr/bin/env python
# -*- coding utf-8 -*-
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  kontakt@markusritschel.de
# Date:   06/09/2021
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#

import logging

import numpy as np


logger = logging.getLogger(__name__)


def cov2corr(cov):
    """Compute the marginal correlations from the covariance matrix.

    \f
    The matrix of the marginal correlations can be written as:
    $$
    R_\text{marg} = D^{-1}\Sigma D^{-1}
    $$
    whereby
    $$
    D = \sqrt{\text{diag}(\Sigma)}
    $$
    is the matrix of the diagonal entries of the rooted variances.
    """
    Σ = cov

    Σ_diag = np.diag(np.diag(Σ))
    Σ_root = np.sqrt(Σ_diag)
    A = np.linalg.inv(Σ_root)

    R = A.dot(Σ).dot(A)

    return R


def prec2partcorr(prec):
    """Compute the partial correlations from the precision matrix."""
    Θ = prec

    Θ_diag = np.diag(np.diag(Θ))
    Θ_root = np.sqrt(Θ_diag)
    A = np.linalg.inv(Θ_root)

    identity_diag = np.ones(Θ.shape) - 2*np.identity(len(Θ))

    R = -A.dot(Θ).dot(A) * identity_diag

    return R


def partcorr(X):
    """Compute the partial correlation matrix from X."""
    Θ = np.linalg.inv(np.cov(X))
    return prec2partcorr(Θ)
