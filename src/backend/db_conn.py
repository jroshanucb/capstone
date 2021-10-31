# from https://towardsdatascience.com/connect-to-postgresql-database-server-using-psycopg2-with-an-elegant-configuration-file-dba6fc885989

from configparser import ConfigParser
import pandas as pd
import psycopg2

# from pathlib import Path
# def get_project_root() -> Path:
#   """Returns project root folder."""
#   return Path(__file__).parents[1]

# Below function reads this section from database.ini file
# [postgresql]
# host = some-host-name
# database = database-name
# user = postgres
# password = password
# port = 5432

def config(config_db):
    section = 'postgresql'
#   config_file_path = 'config/' + config_db
#   database.ini file is in the same folder as this py file
    config_file_path = config_db
    if (len(config_file_path) > 0 and len(section) > 0):
        config_parser = ConfigParser()
        config_parser.read(config_file_path)
        if (config_parser.has_section(section)):
            config_params = config_parser.items(section)
            db_conn_dict = {}
            for config_param in config_params:
                key = config_param[0]
                value = config_param[1]
                db_conn_dict[key] = value
            return db_conn_dict


# Take in a PostgreSQL table and outputs a pandas dataframe
def load_db_table(config_db, query):
    params = config(config_db)
    # print(params)
    engine = psycopg2.connect(**params)
    print("postgres database connection successful")
    data = pd.read_sql(query, con = engine)
    return data

# Take in a PostgreSQL table and outputs a pandas dataframe dictionary compatible using cursors
def run_sql_query(config_db, query):
    params = config(config_db)
    conn = psycopg2.connect(**params)
    print("postgres database connection successful")
    cur = conn.cursor(dictionary=True)
    cur.execute(query)
    conn.commit()
    return True