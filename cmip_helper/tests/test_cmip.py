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
import yaml

from cmip_helper.cmip_proc import list_elements_match_pattern


def test_file_list_fixture(cmip5_files):
    assert type(cmip5_files) == list
    all_files_elements = [len(str(x).split('_')) for x in cmip5_files]
    assert set(all_files_elements) == set([5, 6])


def test_pattern_fixture(cmip5_pattern_dict):
    assert type(cmip5_pattern_dict) == dict


def test_pattern_from_yaml(cmip5_pattern_dict, tmpdir):
    with open(tmpdir / 'cmip5_pattern.yml', 'w') as yaml_file:
        yaml.dump(cmip5_pattern_dict, yaml_file)

    with open(tmpdir / 'cmip5_pattern.yml', 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file)

    assert yaml_dict.keys() == cmip5_pattern_dict.keys()


def test_match_pattern(cmip5_files, cmip5_pattern_dict):
    total_regex = '|'.join([x['regex_pattern'] for x in cmip5_pattern_dict.values()])
    matched, not_matched = list_elements_match_pattern(cmip5_files, total_regex)
    assert len(matched) == len(cmip5_files)
    assert len(not_matched) == 0


@pytest.mark.skip(reason="Not yet implemented")
def test_parse_dir():
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_create_intake_catalog():
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_cmip_parser():
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_validate_files():
    pass
