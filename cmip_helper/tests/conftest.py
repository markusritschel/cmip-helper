#!/usr/bin/env python
# -*- coding utf-8 -*-
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  kontakt@markusritschel.de
# Date:   08/07/2020
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def cmip6_files():
    root_path = Path('/work/cmip6_data/raw/chlos')
    all_file_names = ["chlos_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_185001-20141231.nc"]*100
    all_file_names.extend(["chlos_Omon_MPI-ESM1-2-HR_historical_r1i1p1f1_gn.nc"]*30)
    all_files = list(map(root_path.joinpath, all_file_names))
    return all_files


@pytest.fixture
def cmip5_files():
    root_path = Path("/cmip5/output1/CCCma/CanESM2/historicalGHG/mon/atmos/Amon/r1i1p1/v20111027/tas/")
    all_file_names = ["tas_Amon_CanESM2_historicalGHG_r1i1p1_185001-201212.nc"]*10000
    all_file_names.extend(["tas_Amon_CanESM2_historicalGHG_r1i1p1.nc"]*500)
    all_files = list(map(root_path.joinpath, all_file_names))
    return all_file_names


@pytest.fixture
def cmip5_pattern_dict():
    cmip5_rules = {
        'temporal_file': {
            'filename_pattern': '{variable}_{mip_table}_{model}_{experiment}_{ensemble_member}_{temporal_subset}.nc',
            'regex_pattern': r'([\w-]+_){5}(\d{4,8}-\d{4,8}).nc'
            },
        'gridspec_file': {
            'filename_pattern': '{variable}_{mip_table}_{model}_{experiment}_{ensemble_member}.nc',
            'regex_pattern': r'([\w-]+_){4}[\w]+.nc'
            }
        }
    return cmip5_rules
