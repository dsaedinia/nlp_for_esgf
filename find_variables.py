import json
import sys
from pathlib import Path

import chromadb
import pandas as pd
import spacy
from git import Repo

# from sentence_transformers import SentenceTransformer

# Load global models
nlp = spacy.load("en_core_web_sm")
# sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ACCEPT_PERCENTILE = 0.70
TOP_K_QUERY = 20
TOP_K_FINAL = 10

# Persistent chromaDB
CHROMA_PATH = Path.home() / ".cache" / "NLP_for_ESGF" / "chroma"
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))


def extract_noun_phrases(text: str) -> list[str]:
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks]

    # When adding other facets later with fewer options we will need to extract those phrases out to avoid
    # a situation like monthly soil moisture content being split into "monthly soil moisture content" and not
    # monthly, soil moisture content


def clone_or_update_tables_repo() -> Repo:
    local = Path.home() / ".cache" / "NLP_for_ESGF" / "cmip6-cmor-tables"
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


# def add_standard_name_variant(df) -> pd.DataFrame:
#     """Sometimes the long name isn't helpful and the standard name is better."""
#     df["space_cf_standard"] = df["standard_name"].str.replace("_", " ")
#     return df


def make_all_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Not sure this matters, just in case."""
    df["long_name"] = df["long_name"].str.lower()
    return df


def get_collection(
    df: pd.DataFrame,
    columns: list[str] = ["variable_id", "standard_name", "long_name", "comment"],
):
    """Get or create a chroma collection for variable information."""

    # {"hnsw:space": "cosine"} is a parameter that tells chroma to use cosine similarity for vector comparisons --> ideal for text similarity
    collection = chroma_client.get_or_create_collection(name="variables", metadata={"hnsw:space": "cosine"})
    if collection.count() > 0:
        return collection

    # A document here refers to a variables info cols like long name, standard name, comment etc.
    ids, docs, metas = [], [], []
    for _, row in df.iterrows():
        variable = row["variable_id"]
        for col in columns:
            value = str(row[col]).replace("_", " ").strip()
            if value:
                ids.append(f"{variable}:{col}")
                docs.append(value)
                metas.append({"variable_id": variable})
    collection.add(ids=ids, documents=docs, metadatas=metas)
    return collection


#######################
# Search Funtions     #
#######################
def add_specifity(query: str, df: pd.DataFrame):
    """1 if there is an exact match, 0 if not."""
    tokens = query.split()
    df["specifity"] = (df["variable_id"].isin(tokens) | df["standard_name"].isin(tokens)).astype(bool)
    return df


def find_best_match(
    df: pd.DataFrame,
    phrase: str,
    columns: list[str] = ["variable_id", "standard_name", "long_name", "comment"],
) -> pd.DataFrame:
    df_filtered = df[columns].copy()

    # Specificity --> Exact matches
    df_filtered = add_specifity(phrase, df_filtered)
    if df_filtered["specifity"].any():
        return df_filtered[df_filtered["specifity"]]

    # Semantic embedding and lookup will pull up variable IDs that match
    results = collection.query(query_texts=[phrase], n_results=TOP_K_QUERY)

    # Keep track of variables that we've already seen since the results may have duplicates -->
    # e.g. pr:longname, pr:space_standard_name
    # Results are already sorted by score so first hit will be the "MAX" score
    seen: dict[str, float] = {}
    result_list = zip(results["metadatas"][0], results["distances"][0])
    for meta, dist in result_list:
        variable_id = meta["variable_id"]
        if variable_id not in seen:
            seen[variable_id] = 1 - dist

    top_results = list(seen.keys())[:TOP_K_FINAL]
    df_filtered = df_filtered[df_filtered["variable_id"].isin(top_results)]
    df_filtered["similarity_score"] = df_filtered["variable_id"].map(seen)

    return df_filtered.sort_values("similarity_score", ascending=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        QUERY_STRING = sys.argv[1]
    else:
        QUERY_STRING = "npp"

    # Initialization
    repo = clone_or_update_tables_repo()
    df = create_cv_dataframe(repo)
    df = make_all_lower(df)

    # extract noun phrases
    noun_phrases = extract_noun_phrases(QUERY_STRING)
    # Get or create collection and query for relevant variables based on noun phrases
    collection = get_collection(df)

    # We will loop through each noun phrase and find the best match for each using complementary metrics
    if len(noun_phrases) == 0:
        print("No noun phrases found in the query. Please try a different query.")

    for phrase in noun_phrases:
        df_filtered = find_best_match(df, phrase)
        if not df_filtered.empty:
            print(f"\nTop results for '{phrase}' within our database:")
            print(df_filtered)
