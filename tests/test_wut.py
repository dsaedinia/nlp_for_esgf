import pandas as pd
import pytest

from wut_variable import add_standard_name_variant, extract_noun_phrases


@pytest.fixture
def test_df():
    return pd.DataFrame(dict(standard_name=["sample_name", "another_name"]))


@pytest.fixture
def test_query():
    return "search for minimum air temperature"


def test_add_standard_name_variant(test_df):
    df = add_standard_name_variant(test_df)
    assert ~df["space_cf_standard"].str.contains("_").any()


def test_extract_noun_phrases(test_query):
    noun_phrases = extract_noun_phrases(test_query)
    assert noun_phrases[0] == "minimum air temperature"
