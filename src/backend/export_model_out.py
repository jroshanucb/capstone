from db_conn import load_db_table
from db_conn import config
import pandas as pd
import psycopg2

def db_init():
    config_db = "database.ini"
    params = config(config_db)
    conn = psycopg2.connect(**params)

    return conn

conn = db_init()

query = "select * from model_output"
data = pd.read_sql(query, con = conn)
data.to_csv("model_output.csv", sep=';')
