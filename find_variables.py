import json
import sys
from pathlib import Path

import pandas as pd
import spacy
from git import Repo
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# Load global models
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ACCEPT_PERCENTILE = 0.70
TOP_K_RAPID = 15
TOP_K_FINAL = 10


def extract_noun_phrases(text: str) -> list[str]:
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks]

    # When adding other facets later with fewer options we will need to extract those phrases out to avoid
    # a situation like monthly soil moisture content being split into "monthly soil moisture content" and not
    # monthly, soil moisture content


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

    # Remove duplicate variable_ids, keeping only the first occurrence
    df = df.drop_duplicates(subset="variable_id", keep="first")

    return df


def add_standard_name_variant(df) -> pd.DataFrame:
    """Sometimes the long name isn't helpful and the standard name is better."""
    df["space_cf_standard"] = df["standard_name"].str.replace("_", " ")
    return df


def make_all_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Not sure this matters, just in case."""
    df["long_name"] = df["long_name"].str.lower()
    return df


########################
# Metrics start here
#########################
def add_specifity(query: str, df: pd.DataFrame):
    """1 if there is an exact match, 0 if not."""
    tokens = query.split()
    df["specifity"] = (df["variable_id"].isin(tokens) | df["standard_name"].isin(tokens)).astype(bool)
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


def find_best_match(df: pd.DataFrame, phrase: str):
    df_filtered = df.copy()

    # Specificity --> Exact matches
    df_filtered = add_specifity(phrase, df_filtered)
    if df_filtered["specifity"].any():
        return df_filtered[df_filtered["specifity"]].head(TOP_K_FINAL)

    # Rapidfuzz score -->  Fast metric for fuzzy matching
    df_filtered = add_rapid_score(phrase, df_filtered)
    df_filtered = df_filtered.nlargest(TOP_K_RAPID, "rapid_score")

    # SBERT score rerank --> Slower but more semantically aware metric
    df_filtered = add_SBERT_score(phrase, df_filtered)
    df_filtered = df_filtered.sort_values("max_SBERT_score", ascending=False)

    return df_filtered.head(TOP_K_FINAL)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        QUERY_STRING = sys.argv[1]
    else:
        QUERY_STRING = "search for soil moisture content, tas and minimum air temperature"

    # Initialization
    repo = clone_or_update_tables_repo()
    df = create_cv_dataframe(repo)
    df = make_all_lower(df)
    df = add_standard_name_variant(df)

    # extract noun phrases
    noun_phrases = extract_noun_phrases(QUERY_STRING)
    all_results = {}

    # We will loop through each noun phrase and find the best match for each using complementary metrics
    variables_found = set()
    for phrase in noun_phrases:
        df_filtered = find_best_match(df, phrase)
        print(f"Top matches for '{phrase}':")
        if not df_filtered.empty:
            print(df_filtered)
            variables_found.add((phrase, df_filtered["variable_id"].iloc[0]))
    # printing only top variables found per noun phrase
    # NOTE: we could have multiple variables with tied scores but this is defaulting to one for now
    print("Top variables found across all noun phrases: ")
    for phrase, var in variables_found:
        print(f"{phrase} --> {var}")
