# import sqlite3
import json
import sys
from pathlib import Path

import nltk
import pandas as pd
import spacy
from bert_score import BERTScorer
from git import Repo
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rapidfuzz import fuzz
from scipy.stats import hmean
from sentence_transformers import SentenceTransformer

# Initialize models once globally
bert_scorer = BERTScorer(model_type="bert-base-uncased")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_noun_phrases(text: str) -> list[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks]


def clone_or_update_tables_repo() -> Repo:
    local = Path.home() / ".cache" / "esgf-wut" / "cmip6-cmor-tables"
    if not (local / ".git").exists():
        local.mkdir(exist_ok=True, parents=True)
        repo = Repo.clone_from("https://github.com/PCMDI/cmip6-cmor-tables", local)
    repo = Repo(local)
    repo.remote().update()
    return repo


def _parse_table_json(table_path: Path) -> pd.DataFrame:
    # load json data as a dict
    with open(table_path) as fin:
        data = json.load(fin)
    # if a json conforms to the expected format, it will have this as a key
    if "variable_entry" not in data:
        return pd.DataFrame()
    # we want each variable's dict, but also the key as a 'variable_id'
    df = pd.DataFrame([dict(variable_id=v, **row) for v, row in data["variable_entry"].items()])
    return df


def create_cv_dataframe(repo: Repo) -> pd.DataFrame:
    """Loop through the json files of the target repo and create a dataframe of CV information."""
    df = (
        pd.concat(
            [_parse_table_json(json_path) for json_path in (Path(repo.working_dir) / "Tables").glob("*.json")]
        )
        .fillna("")
        .reset_index(drop=True)
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


def add_specifity(query_list: list[str], df: pd.DataFrame):
    """1 if there is an exact match, 0 if not."""
    df["specifity"] = (df["variable_id"].isin(query_list) | df["standard_name"].isin(query_list)).astype(bool)
    return df


def add_bleu_score(
    query: str,
    df: pd.DataFrame,
    columns: list[str] = ["space_cf_standard", "long_name", "comment"],
):
    df["bleu_score"] = df[columns].apply(
        lambda row: round(
            sentence_bleu(row.to_list(), query, smoothing_function=SmoothingFunction().method1),
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
    nltk.data.path.append("additional_data/nltk_data")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", download_dir="additional_data/nltk_data")

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


# Rapidfuzz implementation
def add_rapid_score(
    query: str,
    df: pd.DataFrame,
    columns: list[str] = ["space_cf_standard", "long_name"],
):
    # for now focussing on cf standard and longname for rapidfuzz --> comments are too noisy
    def rapid_max_token_sort(row):
        scores = []
        for col in columns:
            col_value = str(row[col]).replace("_", " ")
            # fuzz includes a few methods, token_sort_ratio as a start
            # surface air temperature = air surface temperature treated the same
            score = fuzz.token_sort_ratio(query.lower(), col_value)
            scores.append(score)
        return round(max(scores) / 100, 4)

    df["rapid_score"] = df[columns].apply(rapid_max_token_sort, axis=1)
    return df


def add_BERT_score(
    query: str, df: pd.DataFrame, columns: list[str] = ["space_cf_standard", "long_name", "comment"]
) -> pd.DataFrame:
    """
    Add BERTScore column to dataframe --> Allows for semantic similarity, but may require extra processing
    """

    def max_BERT_score(row):
        scores = []
        for col in columns:
            col_value = str(row[col]).replace("_", " ")
            if col_value:
                P, R, F1 = bert_scorer.score([query], [col_value])
            else:
                scores.append(0.0)
                continue
            scores.append(F1.mean().item())
        return round(max(scores), 4)

    df["max_BERT_score"] = df[columns].apply(max_BERT_score, axis=1)
    return df


# SBERT is similar to BERT but optimized for sentence similarity
# Key point being it is faster
def add_SBERT_score(
    query: str, df: pd.DataFrame, columns: list[str] = ["space_cf_standard", "long_name", "comment"]
) -> pd.DataFrame:
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)

    def max_SBERT_score(row):
        scores = []
        for col in columns:
            col_value = str(row[col]).replace("_", " ")
            # sometimes the value is an empty string and some metrics don't like that
            if col_value:
                col_embedding = sbert_model.encode(col_value, convert_to_tensor=True)
                similarity_score = sbert_model.similarity(query_embedding, col_embedding).item()
                scores.append(similarity_score)
            else:
                scores.append(0.0)
        return round(max(scores), 4)

    df["max_SBERT_score"] = df[columns].apply(max_SBERT_score, axis=1)
    return df


def add_harmonic_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Add a harmonic mean of the given columns"""
    df["hmean_score"] = df.apply(
        lambda row: hmean([row[col] for col in columns]),
        axis=1,
    )
    return df


if __name__ == "__main__":
    ACCEPT_PERCENTILE = 0.95
    if len(sys.argv) > 1:
        QUERY_STRING = sys.argv[1]
    else:
        QUERY_STRING = "search for minimum air temperature and maximum air temperature"

    # Initialization
    repo = clone_or_update_tables_repo()
    df = create_cv_dataframe(repo)
    df = make_all_lower(df)
    df = add_standard_name_variant(df)

    # extract noun phrases from query
    noun_phrases = extract_noun_phrases(QUERY_STRING)
    print(noun_phrases)
    for phrase in noun_phrases:
        print(f"Noun phrase: {phrase}")
        # Add scores
        df = add_specifity(phrase.split(), df)
        df = add_rapid_score(phrase, df)
        df = add_bleu_score(phrase, df)
        df = add_meteor_score(phrase.split(), df)

        # Adding semantic based scores using embeddings - heavier cost
        # BERT adds a significant amount of time
        df = add_BERT_score(phrase, df)
        # SBERT should be quicker than BERT
        df = add_SBERT_score(phrase, df)

        df = add_harmonic_mean(
            df, columns=["bleu_score", "rapid_score", "max_meteor_score", "max_SBERT_score"]
        )

        # If the input phrase uses an exact match of the CV, get rid of everything else
        if df["specifity"].max():
            df = df[df["specifity"]]

        # Let's try a filter where we show only the dataframe that lies in the top
        # 10% of at least one of the scores
        desc = df.describe(percentiles=[ACCEPT_PERCENTILE])
        score_columns = [col for col in df.columns if "score" in col]
        df_filtered = df[
            df.apply(
                lambda row: any(
                    row[col] > desc.loc[f"{int(round(ACCEPT_PERCENTILE * 100))}%", col]
                    for col in score_columns
                ),
                axis=1,
            )
        ]
        print(
            df_filtered.sort_values(
                [
                    "hmean_score",
                    "bleu_score",
                    "rapid_score",
                    "max_meteor_score",
                    "max_SBERT_score",
                ],
                ascending=False,
            )
        )
