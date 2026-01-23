# import sqlite3
import json
import sys
from pathlib import Path
import pandas as pd
from git import Repo
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from scipy.stats import hmean
from rapidfuzz import fuzz


def clone_or_update_tables_repo() -> Repo:
    local = Path.home() / ".cache" / "esgf-wut" / "cmip6-cmor-tables"
    if not (local / ".git").exists():
        local.mkdir(exist_ok=True, parents=True)
        repo = Repo.clone_from("https://github.com/PCMDI/cmip6-cmor-tables", local)
    repo = Repo(local)
    repo.remote().update()
    return repo


def parse_table_json(table_path: Path) -> pd.DataFrame:
    # load json data as a dict
    with open(table_path) as fin:
        data = json.load(fin)
    # if a json conforms to the expected format, it will have this as a key
    if "variable_entry" not in data:
        return pd.DataFrame()
    # we want each variable's dict, but also the key as a 'variable_id'
    df = pd.DataFrame(
        [dict(variable_id=v, **row) for v, row in data["variable_entry"].items()]
    )
    return df


def add_standard_name_variant(df) -> pd.DataFrame:
    """Sometimes the long name isn't helpful and the standard name is better."""
    df["space_cf_standard"] = df["standard_name"].str.replace("_", " ")
    return df


def make_all_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Not sure this matters, just in case."""
    df["long_name"] = df["long_name"].str.lower()
    return df


def add_specifity_score(query_list: list[str], df: pd.DataFrame):
    """1 if there is an exact match, 0 if not."""
    df["specifity_score"] = (
        df["variable_id"].isin(query_list) | df["standard_name"].isin(query_list)
    ).astype(float)
    return df


def add_bleu_score(
    query: str,
    df: pd.DataFrame,
    columns: list[str] = ["space_cf_standard", "long_name", "comment"],
):
    df["bleu_score"] = df[columns].apply(
        lambda row: round(
            sentence_bleu(
                row.to_list(), query, smoothing_function=SmoothingFunction().method1
            ),
            4,
        ),
        axis=1,
    )
    return df


def add_meteor_score(
    query: list[str],
    df: pd.DataFrame,
    columns: list[str] = ["space_cf_standard", "long_name", "comment"],
):
    # Need to specify path for NLTK because that is where it will check and see if corpora exists
    nltk.data.path.append("database/nltk_data")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", download_dir="database/nltk_data")

    # This will add only the highest score it can find among the cols
    def max_meteor_score(row):
        scores = []
        for col in columns:
            col_value = str(row[col]).replace("_", " ").split()
            score = meteor_score([col_value], query)
            scores.append(score)
        return round(max(scores), 4)

    # Will leave out for now
    # def combined_meteor_score(row):
    #     combined_text = " ".join(
    #         str(row[col]).replace("_", " ") for col in columns
    #     ).split()
    #     return round(meteor_score([combined_text], query), 4)

    # may add a weighted score --> can weigh cols differently
    # Say long_name is more important than comment, etc.

    df["max_meteor_score"] = df[columns].apply(max_meteor_score, axis=1)
    # df["combined_meteor_score"] = df[columns].apply(combined_meteor_score, axis=1)
    return df


# still experimenting with rapidfuzz
def rapid_score(
    query: str,
    df: pd.DataFrame,
    columns: list[str] = ["space_cf_standard", "long_name"],
):
    # for now focussing on cf standard and longname for rapidfuzz
    def rapid_max_token_sort(row):
        scores = []
        for col in columns:
            col_value = str(row[col]).replace("_", " ")
            # fuzz includes several methods, token_sort_ratio as a start
            # surface air temperature = air surface temperature treated the same
            score = fuzz.token_sort_ratio(query.lower(), col_value)
            scores.append(score)
        return round(max(scores) / 100, 4)

    df["rapid_score"] = df[columns].apply(rapid_max_token_sort, axis=1)
    return df


if __name__ == "__main__":
    BLEU_THRESHOLD = 0.7
    if len(sys.argv) > 1:
        QUERY_STRING = sys.argv[1]
    else:
        QUERY_STRING = "air temperature"

    repo = clone_or_update_tables_repo()
    df = (
        pd.concat(
            [
                parse_table_json(json_path)
                for json_path in (Path(repo.working_dir) / "Tables").glob("*.json")
            ]
        )
        .fillna("")
        .reset_index()
    )

    # print(df[["variable_id", "standard_name", "long_name", "comment"]])

    df = make_all_lower(df)
    df = add_standard_name_variant(df)

    df = add_specifity_score(QUERY_STRING.split(), df)
    if df["specifity_score"].max() > 0.99:
        df = df[df["specifity_score"] > 0.99]

    df = rapid_score(QUERY_STRING, df)
    df = add_bleu_score(QUERY_STRING, df)
    df = add_meteor_score(QUERY_STRING.split(), df)

    print("Sort by bleu score:")
    print(df.sort_values("bleu_score", ascending=False))

    print("Sort by max meteor score:")
    print(df.sort_values("max_meteor_score", ascending=False))

    # print("Sort by combined meteor score:")
    # print(df.sort_values("combined_meteor_score", ascending=False))

    print("Sort by rapidfuzz score:")
    print(df.sort_values("rapid_score", ascending=False))

    # add harmonic mean - unsure of this will be helpful as of now
    df["hmean_score"] = df.apply(
        lambda row: hmean([row["bleu_score"], row["max_meteor_score"]]),
        axis=1,
    )
    print(df.sort_values("hmean_score", ascending=False))
