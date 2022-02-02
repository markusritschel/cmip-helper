#!/usr/bin/env python
# -*- coding utf-8 -*-
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Author: Markus Ritschel
# eMail:  kontakt@markusritschel.de
# Date:   03/07/2020
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
import json
import logging
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import compress
from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
import xarray as xr
from cmip6_preprocessing.preprocessing import combined_preprocessing
from intake.source.utils import reverse_formats

try:
    from cdo import *
    cdo = Cdo()
except:
    pass

__all__ = ['list_elements_match_pattern', 'create_intake_catalog', 'parse_dir', 'parse_file_storage', 'slice_picontrol', 'pd_read_member']


logger = logging.getLogger(__name__)


def list_elements_match_pattern(input_list, regex):
    """Check a list of strings/files for matching a regular expression.

    Parameters
    ----------
    input_list : list
        A list of strings (e.g. file paths) that will be checked agains the regular expression.
    regex : str
        A string holding a valid regular expression that will be used to filter the INPUT_LIST.

    Returns
    -------
    matched : list
        A list of elements from INPUT_LIST that match the regular expression.
    not_matched : list
        A list of elements from INPUT_LIST that do not match the regular expression.

    Examples
    --------
    >>> l = ['']
    >>> regex = r""
    >>> list_elements_match_pattern()
    # TODO: complete this example!
    """
    matches = list(map(re.compile(regex).fullmatch, map(str, input_list)))
    not_matched = list(compress(input_list, [x is None for x in matches]))
    matched = list(compress(input_list, [x is not None for x in matches]))
    return matched, not_matched


def create_intake_catalog(path, cmip_version=None, name=None, rules=None, **kwargs):
    """Create a pair of two files (a collection JSON file and a catalog CSV.GZ file), which can be opened by `intake`.
    # TODO: complete this docstring with Parameters/Returns/Examples
    """
    if (not name) & (cmip_version is not None):
        name = f"cmip{cmip_version}_catalog"
    elif (not name) & (not cmip_version):
        # logging.info("Didn't specify name nor CMIP version")
        name = 'my_catalog'

    path = Path(path).expanduser()
    cat_file = path/f"{name}.csv.gz"
    json_file = path/f"{name}.json"
    print("Create {} and {} in {}".format(*map(os.path.basename, [json_file, cat_file]), path))

    df = parse_dir(path, cmip_version=cmip_version, **kwargs)
    df.to_csv(cat_file, compression='gzip', index=False)

    attributes = [{"column_name": name, "vocabulary": ""} for name in df.columns if not name == 'path']
    json_dict = {
        "esmcat_version": "0.1.0",
        "id": name,
        "description": f"This is an ESM collection for CMIP{cmip_version} data",
        "catalog_file": str(cat_file),
        "attributes": attributes,
        "assets": {
            "column_name": "path",
            "format": "netcdf"
            },
        "aggregation_control": {
            "variable_column_name": "variable_id",
            "groupby_attrs": [
                "source_id",
                "experiment_id",
                # "table_id",
                # "grid_label"
                ],
            "aggregations": [
                {
                    "type": "union",
                    "attribute_name": "variable_id"
                    },
                {
                    "type": "join_existing",
                    "attribute_name": "time_range",
                    "options": {"dim": "time", "coords": "minimal", "compat": "override"}
                    },
                {
                    "type": "join_new",
                    "attribute_name": "member_id",
                    "options": {"coords": "minimal", "compat": "override"}
                    }
                ]
            }
        }

    with open(str(json_file), 'w') as jf:
        json.dump(json_dict, jf, indent=2)

    print("\nYou can load the catalog with")
    print(f""">>> col_file = "{json_file}"\n>>> intake.open_esm_datastore(col_file)""")

    return str(json_file)


def parse_dir(path, depth=12, ext='*.nc', **kwargs):
    """Retrieve all netCDF under a certain path (including sub-directories and parse their names according to a
    given pattern (`file_fmt`). Returns a pd.DataFrame of the parsed elements.
    # TODO: complete this docstring with Parameters/Returns/Examples
    """
    path = Path(path).expanduser()

    # files = all_files = path.rglob('*.nc')
    # files = [x.name for x in all_files]
    cmd = f"find {path.as_posix()}/ -maxdepth {depth} -iname '{ext}'"
    find_res = subprocess.run(cmd, shell=True, capture_output=True)
    all_files = find_res.stdout.decode('utf-8').split()
    # files = all_files = glob.glob(os.path.join(path, '**/*.nc'), recursive=True)
    dirs, files = zip(*map(os.path.split, all_files))
    # files, ext = zip(*map(os.path.splitext, files))

    # guess version by 1st file
    cmip_version = kwargs.pop('cmip_version', None)
    if 'file_fmt' in kwargs:
        file_fmt = kwargs.pop('file_fmt')
    else:
        parser = CMIPparser(cmip_version, guess_by=all_files[0])
        file_fmt = parser.filename_template

    # TODO: try-except? Or how can you filter out those files that don't match file_fmt?
    # TODO: how to take into account gridspec files and normal temporal files in the same directory?
    # TODO: Check cmip.py at https://github.com/NCAR/intake-esm-datastore/blob/master/builders/cmip.py
    rev_dict = reverse_formats(file_fmt, files)
    rev_dict['path'] = all_files

    return pd.DataFrame.from_dict(rev_dict)


def parse_file_storage(path, depth=1, **kwargs):
    # TODO: create a docstring with Parameters/Returns/Examples
    cat = FileCatalog()
    print(path)
    cat.populate_from_directory(path, depth=depth, **kwargs)
    return cat


def cdo_multiproc(cdo_cmd, src_dir, output_dir, retain_structure=True):
    # TODO: create a docstring with Parameters/Returns/Examples
    tasks = []
    nprocs = os.cpu_count()//3

    # input_files = get all files below src_dir level
    # output_files = output_dir / input_dir_struct / filenames

    # cdo_cmd = cdo_cmd + input_files + output_files

    with ProcessPoolExecutor(nprocs) as pool:
        task = pool.submit(subprocess.run, cdo_cmd, shell=True, capture_output=True)

    return


def xr_read_ensemble(df, **kwargs):
    # TODO: create a docstring with Parameters/Returns/Examples
    _id = kwargs.pop('group_id', None)
    df = df.sort_values('member_id')

    files = df.path.values

    # files = [cdo.yearmin(input=file) for file in files]

    logger.info('%s: open_mfdataset', _id)
    ds = xr.open_mfdataset(files, chunks={'time': 12}, concat_dim='member',
                           # parallel=True,
                           combine='nested',
                           # data_vars='minimal',
                           coords='minimal',
                           compat='override',
                           preprocess=combined_preprocessing,
                           # decode_times=True,
                           # use_cftime=True,
                           )

    logger.info('%s: assign_coord member', _id)
    mem_ids_dim = xr.DataArray(df.member_id.to_list(), dims=['member'])
    ds = ds.assign_coords({'member': mem_ids_dim})

    # ValueError: Codec does not support buffers of > 2147483647 bytes
    #     ds = ds.chunk({'time':12*100})
    #     logger.info('%s: apply wrapper', _id)
    #     ds = wrapper(ds)
    #     ds = ds.chunk({'time':12*100, 'x':50,'y':50})
    # #     ds = ds.unify_chunks()

    #     logger.info('%s: encoding = {}', _id)
    #     # workaround according to https://github.com/pydata/xarray/issues/2300#issuecomment-598790404
    #     ds.encoding = {}
    #     for var in ds.variables:
    #         ds[var].encoding = {}

    logger.info('%s: apply wrapper', _id)
    # ds = combined_preprocessing(ds)

    ds = _squeeze_coords(ds)

    return ds


def pd_read_member2(file):
    # TODO: create a docstring with Parameters/Returns/Examples
    ds = xr.open_dataset(file, chunks={}, decode_times=True, decode_cf=True, use_cftime=True)
    ds = combined_preprocessing(ds)
    df = ds.tas.squeeze().to_pandas()
    #     df.index = df.index.to_period('M')
    df.index = df.index.to_datetimeindex().to_period('M')
    print('done reading')
    # df = df.rename(f"i={i}")
    return df


def pd_read_member(file, variable):
    """Read a single member of a model ensemble (i.e. a single netCDF file), select the variable and transform it to a
    :meth:`pandas.DataFrame`.
    # TODO: create a docstring with Parameters/Returns/Examples
    """
    ds = xr.open_dataset(file, chunks={}, decode_times=True, decode_cf=True, use_cftime=True)
    df = ds[variable].squeeze()
    ds = df.to_pandas()
    # ds = combined_preprocessing(ds)
    # ds = ds['siarea'].squeeze()
    return ds


def _squeeze_coords(ds):
    """Squeeze an :meth:`xarray.Dataset` object
    # TODO: create a docstring with Parameters/Returns/Examples"""
    if ('lon' in ds.dims) and ('lat' in ds.dims):
        ds = ds.squeeze(['lon', 'lat'], drop=True)
    elif ('x' in ds.dims) and ('y' in ds.dims):
        ds = ds.squeeze(['x', 'y'], drop=True)
    return ds


class FileCatalog(object):
    # TODO: create a docstring with Parameters/Returns/Examples
    def __init__(self, key_template=None, file_fmt=None):
        self.df = None
        self.key_template = key_template or 'source_id.grid_label.member_id'
        self._group_elements = self.key_template.split('.') # ['source_id', 'variable_id', 'grid_label', 'member_id']
        # self.key_template = '.'.join(self._group_elements)
        self.categories = None

    def populate_from_directory(self, path, depth=1, **kwargs):
        self.df = parse_dir(path, depth=depth, **kwargs)

        file_fmt = kwargs.pop("file_fmt", None)
        if file_fmt:
            files = self.df.path.values
            rev_dict = reverse_formats(file_fmt, files)
            self._group_elements = rev_dict.keys()

        self.categories = self.df[self._group_elements]
        self.categories.index = self.categories.T.apply('.'.join)

    def to_dataset_dict(self):
        tasks = []
        nprocs = os.cpu_count()//3

        with ProcessPoolExecutor(nprocs) as pool:
            print('Add jobs to process pool... ', end='')
            for group_id, subdf in self.df.groupby(['source_id', 'variable_id', 'grid_label']):
                if isinstance(group_id, tuple):
                    group_id = '.'.join(group_id)

                # # FGOALS raus
                # if group_id.startswith('FGOALS'):
                #     continue
                task = pool.submit(xr_read_ensemble, subdf, group_id=group_id)
                task.id = group_id
                tasks.append(task)
            print('done')

            print('Process files... ')
            ds_dict = {}
            for task in tqdm(as_completed(tasks), total=len(tasks), leave=True):
                logger.info("finished %s", task.id)
                ds_dict[task.id] = task.result()

        return ds_dict

    def to_dataframe(self, frequency='M'):
        """Processes netCDF files containing a 1D time series and merges them into a single pd.DataFrame"""
        tasks = []
        nprocs = os.cpu_count()//3

        # print("No Multiproc")
        # out = []
        # for i in tqdm(range(len(self.df))):
        #     subdf = self.df.iloc[i]
        #     ds = xr.open_dataset(subdf.path, chunks={}, decode_times=True, decode_cf=True, use_cftime=True)
        #     id = '.'.join(subdf[self._group_elements].values)
        #     variable = subdf['variable_id']
        #     df = ds[variable].squeeze().to_pandas()
        #     freq = frequency
        #     df.index = df.index.to_datetimeindex().to_period(freq)
        #     df = df.rename(id)
        #     out.append(df)
        # return pd.concat(out, join='outer', axis=1)



        with ProcessPoolExecutor(nprocs) as pool:
            logger.debug('Add jobs to process pool... ')
            for i in range(len(self.df)):
                subdf = self.df.iloc[i]
                task = pool.submit(pd_read_member, subdf.path, subdf['variable_id'])
                task.id = '.'.join(subdf[self._group_elements].values)
                task.variable = subdf['variable_id']
                # if task.variable.startswith('siarea'):
                #     task.variable = task.variable[:-1]
                tasks.append(task)
            logger.debug('done')

            out = []
            for task in tqdm(as_completed(tasks), total=len(tasks)):
                df = task.result()
                if not isinstance(df, (pd.Series, pd.DataFrame)):
                    logger.warning("Instance %s is not a Pandas object. Skip entry.", task.id)
                    continue
                # df = df[task.variable].squeeze()
                # df = df.to_pandas()
                #     df.index = df.index.to_period('M')
                # df.index = df.index.to_datetimeindex().to_period('M')  # TODO: redundant?
                # freq = df.index.inferred_freq
                freq = frequency
                # restrict the time limit for future scenarios since 2262 is the latest year Pandas can deal with
                # according to https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units
                df = df[df.index < pd.to_datetime('2262-01-01 00:00:00')]
                df.index = df.index.to_datetimeindex().to_period(freq)
                # df = df.rename(task.id)
                df = df.to_frame(name=task.id)
                df = df[~df.index.duplicated(keep='first')]
                out.append(df)
            df = pd.concat(out, join='outer', axis=1)
            print(type(df))

        # print("INIT COLKEYS")
        # df.ens.init_colkeys(self.key_template)
        return df


class CMIPparser():
    # TODO: create a docstring with Parameters/Returns/Examples
    def __init__(self, version, guess_by=None):
        self.version = str(version)
        self.guess_by_file = guess_by
        self._validate()

    # @staticmethod
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
        if self.version == '5':
            return '{variable_id}_{table_id}_{source_id}_{experiment_id}_{member_id}_{time_range}.nc'
        elif self.version == '6':
            return '{variable_id}_{table_id}_{source_id}_{experiment_id}_{member_id}_{grid_label}_{time_range}.nc'
        else:
            return None

    @property
    def gridspec_template(self):
        if self.version == '5':
            return '{variable_id}_{table_id}_{source_id}_{experiment_id}_{member_id}.nc'
        elif self.version == '6':
            return '{variable_id}_{table_id}_{source_id}_{experiment_id}_{member_id}_{grid_label}.nc'
        else:
            return None

    # @staticmethod
    def _guess_version(self):
        elements = self.guess_by_file.split('_')
        if re.match(r"\d{4,8}-\d{4,8}.nc$", elements[-1]):
            file_type = 'temporal'
            if len(elements) == 6:
                self.version = '5'
            elif len(elements) == 7:
                self.version = '6'
        elif re.match(r"r\d+?i\d+?p\d+?.nc$", elements[-1]) or re.match(r"[a-z]{1,2}.nc$", elements[-1]):
            file_type = 'gridspec'
            if len(elements) == 5:
                self.version = '5'
            elif len(elements) == 6:
                self.version = '6'

        return


@pd.api.extensions.register_dataframe_accessor("ens")
class PandasEnsembleAccessor:
    # TODO: create a docstring with Parameters/Returns/Examples
    def __init__(self, pandas_obj):
        # self._validate(pandas_obj)
        self._df = pandas_obj
        # self.column_keys = None

    @staticmethod
    def _validate(obj):
        """Just an example"""
        # verify there is a column latitude and a column longitude
        if 'latitude' not in obj.columns or 'longitude' not in obj.columns:
            raise AttributeError("Must have 'latitude' and 'longitude'.")

    def groupby(self, cat_map):
        return self._df.groupby(self._df.columns.map(cat_map), axis=1)

    def init_colkeys(self, key_template, sep='.'):
        """Split the column names of the :meth:`pandas.DataFrame` object according to the key_template and creates a new
        :meth:`pandas.DataFrame` containing the resulting elements column-wise with the names of the key_template as
        column names and the original column names as index.

        Parameters
        ----------
        key_template : str
            String containing the column keys according to which the pd.DataFrame shall be grouped.
            This can look like "experiment_id.source_id.member_id".
        sep : str
            The separator which will be used for splitting the column names. Make sure the key_template contains these separators.
        """
        if not sep in key_template:
            raise ValueError("key_template must contain at least one occurrence of the separator.")

        def all_keys_equal():
            len_of_colkeys = [len(x.split(sep)) for x in self._df.columns]
            return len(set(len_of_colkeys)) == 1

        if not all_keys_equal():
            raise ValueError("Column keys must show the same pattern. Not all column names have the same number of keys.")

        self.column_keys = pd.DataFrame(pd.Series(self._df.columns).str.split(sep, expand=True))
        self.column_keys.index = self._df.columns
        self.column_keys.columns = key_template.split(sep)
        return

    def groupbycolkey(self, key):
        """Groups the pd.DataFrame by column key. Column keys must be initialized beforehand."""
        if self.column_keys is None:
            raise AttributeError("Ensure that the column key template is initialized using `init_colkeys(key_template)`.")
        return self._df.groupby(self.column_keys[key], axis=1)


@xr.register_dataset_accessor("ens")
class XarrayEnsembleAccessor:
    """An xarray.Dataset accessor supporting the grouping of ensemble members by model id and similar.
    The `member` coordinate in the xr.Dataset must have a `key_template` attribute of the form 'source_id.member_id.grid_label',
    following the structure of the entries of the 'member' coordinate."""
    xr.set_options(keep_attrs=True)

    def __init__(self, xarray_obj):
        self._ds = xarray_obj
        self._validate(xarray_obj)
        self.init_member_key()

    @property
    def key_template(self):
        """Return the key template"""
        return self._ds.coords['member'].attrs['key_template']

    @staticmethod
    def _validate(obj):
        """Test the xarray.Dataset for the existance of the coordinate 'member' and its attribute 'key_template'.
        This routine is run as soon as the accessor is called."""
        if not 'member' in obj.coords:
            raise AttributeError("No coordinate 'member' found in xarray object.")
        elif not 'key_template' in obj.coords['member'].attrs:
            logger.warning("No 'key_template' found in attribute list of 'member' coordinate.")
        return

    def init_member_key(self, **kwargs):
        """Initialize a helper object (xarray.DataArray) that holds the elements of each member as variables, following the key_template,
        and the member name itself as coordinate. This serves as a basis for the grouping later."""
        key_template = kwargs.pop('key_template', self.key_template)
        sep = kwargs.pop('sep', '.')

        if not sep in key_template:
            raise ValueError("key_template must contain at least one occurrence of the separator.")

        def _all_keys_equal():
            len_of_colkeys = [len(x.split(sep)) for x in self._ds.member.values]
            return len(set(len_of_colkeys)) == 1

        if not _all_keys_equal():
            raise ValueError("Column keys must show the same pattern. Not all column names have the same number of keys.")

        self.member_keys = pd.DataFrame(pd.Series(self._ds.member.values).str.split(sep, expand=True))
        self.member_keys.index = self._ds.member.values
        self.member_keys.index.name = 'member'
        column_names = key_template.split(sep)
        if not len(column_names) == len(self.member_keys.columns):
            raise ValueError("It seems like the key_template does not fit the structure of the member ids in the 'member' coordinate.")
        self.member_keys.columns = column_names
        self.member_keys = self.member_keys.to_xarray()
        return


    def groupby(self, key):
        """Group the xarray.Dataset by a member key. The key is an element of the key_template. For example, if the members
        have the format 'ACCESS-CM2.r1i1p1f1.gn', then the key_template should be 'source_id.member_id.grid_label'. The dataset
        can then be grouped, for example, via `ds.ens.groupby('source_id')`."""
        if self.member_keys is None:
            raise AttributeError("Ensure that the column key template is initialized using `init_colkeys(key_template)`.")
        return self._ds.groupby(self.member_keys[key])


def slice_picontrol():
    """piControl data are usually not existent as single runs but as one long time series.
    This splits the long time series into several files of equal length and with some overlap (see old cmip5-bayes routines)"""
    pass
