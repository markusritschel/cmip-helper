#!/usr/bin/env python
# -*- coding utf-8 -*-
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  kontakt@markusritschel.de
# Date:   03/07/2020
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import glob
import json
import logging
import os
import re
from pathlib import Path
from functools import partial
from itertools import compress
import pandas as pd
from intake.source.utils import reverse_formats


def list_elements_match_pattern(files, regex):
    """Check a list of strings/files for matching a regular expression"""
    matches = list(map(re.compile(regex).fullmatch, map(str, files)))
    not_matched = list(compress(files, [x is None for x in matches]))
    matched = list(compress(files, [x is not None for x in matches]))
    return matched, not_matched


def create_intake_catalog(path, version=None, name=None, rules=None):
    """Creates a pair of two files (a collection JSON file and a catalog CSV.GZ file),
    which can be opened by `intake`"""
    if (not name) & version:
        name = f"cmip{version}_catalog"
    else:
        # logging.info("Didn't specify name nor CMIP version")
        name = 'my_catalog'

    path = Path(path)
    cat_file = path / f"{name}.csv.gz"
    json_file = path / f"{name}.json"
    print("Create {} and {} in {}".format(*map(os.path.basename, [json_file, cat_file]), path))

    df = parse_dir(path)
    df.to_csv(cat_file, compression='gzip', index=False)

    attributes = [{"column_name": name, "vocabulary": ""} for name in df.columns if not name=='path']
    json_dict = {
        "esmcat_version":      "0.1.0",
        "id":                  name,
        "description":         f"This is an ESM collection for CMIP{version} data",
        "catalog_file":        cat_file,
        "attributes":          attributes,
        "assets": {
            "column_name": "path",
            "format":      "netcdf"
            },
        "aggregation_control": {
            "variable_column_name": "variable_id",
            "groupby_attrs": [
                "experiment_id",
                "source_id",
                ],
            "aggregations": [
                {
                    "type":           "union",
                    "attribute_name": "variable_id"
                    },
                {
                    "type":           "join_existing",
                    "attribute_name": "time_range",
                    "options":        {"dim": "time", "coords": "minimal", "compat": "override"}
                    },
                {
                    "type":           "join_new",
                    "attribute_name": "member_id",
                    "options":        {"coords": "minimal", "compat": "override"}
                    }
                ]
            }
        }

    with open(json_file, 'w') as jf:
        json.dump(json_dict, jf, indent=2)

    print("\nYou can load the catalog with")
    print(f""">>> col_file = "{json_file}"\n>>> intake.open_esm_datastore(col_file)""")

    return json_file


def parse_dir(path, **kwargs):
    """Retrieve all netCDF under a certain path (including sub-directories and parse their names according to a
    given pattern (`file_fmt`). Returns a pd.DataFrame of the parsed elements.
    """
    path = os.path.expanduser(path)

    files = all_files = glob.glob(os.path.join(path, '**/*.nc'), recursive=True)
    dirs, files = zip(*map(os.path.split, all_files))
    #     files, ext = zip(*map(os.path.splitext, files))

    # guess version by 1st file
    cmip_version = kwargs.pop('cmip_version', None)
    parser = CMIPparser(cmip_version, guess_by=files[0])
    file_fmt = parser.filename_template

    # TODO: try-except? Or how can you filter out those files that don't match file_fmt?
    # TODO: how to take into account gridspec files and normal temporal files in the same directory?
    # TODO: Check cmip.py at https://github.com/NCAR/intake-esm-datastore/blob/master/builders/cmip.py
    rev_dict = reverse_formats(file_fmt, files)
    rev_dict['path'] = all_files

    return pd.DataFrame.from_dict(rev_dict)


class CMIPparser():
    def __init__(self, version, guess_by=None):
        self.version = str(version)
        self.guess_by_file = guess_by
        self._validate()

    @staticmethod
    def _validate(self):
        if self.version in ['5', '6']:
            return True
        elif self.guess_by_file:
            print('Version is not in [5, 6]. Try guessing cmip version by a file instead.')
            self.guess_by_file = Path(self.guess_by_file)
            if not self.guess_by_file.is_file():
                raise KeyError("guess_by parameter must point to a valid file")
            self._guess_version()
        else:
            raise ValueError(
                "Either version [5, 6] must be given or hand a file to the parameter `guess_by` to guess the version")

    @property
    def filename_template(self):
        if self.version=='5':
            return '{variable}_{mip_table}_{model}_{experiment}_{ensemble_member}_{temporal_subset}.nc'
        elif self.version=='6':
            return '{variable_id}_{table_id}_{source_id}_{experiment_id}_{member_id}_{grid_label}_{time_range}.nc'
        else:
            return None

    @property
    def gridspec_template(self):
        if self.version=='5':
            return '{variable}_{mip_table}_{model}_{experiment}_{ensemble_member}.nc'
        elif self.version=='6':
            return '{variable_id}_{table_id}_{source_id}_{experiment_id}_{member_id}_{grid_label}.nc'
        else:
            return None

    @staticmethod
    def _guess_version(self):
        elements = self.guess_by_file.split('_')
        if re.match(r"\d{4,8}-\d{4,8}.nc$", elements[-1]):
            file_type = 'temporal'
            if len(elements)==6:
                self.version = '5'
            elif len(elements)==7:
                self.version = '6'
        elif re.match(r"r\d+?i\d+?p\d+?.nc$", elements[-1]) or re.match(r"[a-z]{1,2}.nc$", elements[-1]):
            file_type = 'gridspec'
            if len(elements)==5:
                self.version = '5'
            elif len(elements)==6:
                self.version = '6'

        return


def slice_picontrol():
    """piControl data are usually not existent as single runs but as one long time series.
    This splits the long time series into several files of equal length and with some overlap (see old cmip5-bayes routines)"""
    pass