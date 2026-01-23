import sqlite3
import requests_cache
import backoff
import requests
from requests_cache import NEVER_EXPIRE


@backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_tries=5,
                      on_backoff=lambda details: print(
                          f"Retrying {details['target']}: Try #{details['tries']}"
                      )
                      )
def get_data(url: str) -> dict:
    session = requests_cache.CachedSession(
        'esgf_request_cache', expire_after=NEVER_EXPIRE)
    response = session.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    return response.json()


def get_max_facet(facets: list) -> str:
    max_facet = ""
    max_num = 0
    # this is dependent on the list always being even and structured like this ['Air Temperature', 210147, 'Air Temp', 9]
    for i in range(0, len(facets) - 1, 2):
        # count after actual facet
        if facets[i + 1] > max_num:
            max_num = facets[i + 1]
            max_facet = facets[i]
    return max_facet


def create_table(db_path: str, facets: tuple):
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")
    cursor.execute("""
CREATE TABLE IF NOT EXISTS Variables(
variable_id  TEXT PRIMARY KEY,
variable_long_name TEXT NOT NULL,
variable_units TEXT NOT NULL,
cf_standard_name TEXT NOT NULL                 
);""")

    cursor.execute(
        f"INSERT INTO Variables VALUES {facets} ON CONFLICT DO NOTHING")
    connection.commit()

    cursor.close()
    connection.close()


def build_variable_database():
    # going to grab all the variable_ids first
    facets = get_data(
        "https://esgf-node.ornl.gov/esgf-1-5-bridge/?project=CMIP6&limit=0&facets=variable_id")
    variable_ids = facets["facet_counts"]["facet_fields"]["variable_id"][::2]
    # then loop through and make request with each individual one
    for v in variable_ids:
        data = get_data(
            f"https://esgf-node.ornl.gov/esgf-1-5-bridge/?project=CMIP6&limit=0&variable_id={v}&facets= variable_units, variable_long_name, cf_standard_name")

        facet_fields = data["facet_counts"]["facet_fields"]

        long_name = get_max_facet(facet_fields["variable_long_name"])
        v_units = get_max_facet(facet_fields["variable_units"])
        cf_standard = get_max_facet(facet_fields["cf_standard_name"])

        print((v, long_name, v_units, cf_standard))
        # create the table and pass the tuple
        create_table("cmip6_variables.db",
                     (v, long_name, v_units, cf_standard))


build_variable_database()
