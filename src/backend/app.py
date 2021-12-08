from typing import Sequence
import flask
from flask import request, jsonify
from flask_cors import CORS
from random import seed
from random import choice
from db_conn import load_db_table
import json
import random
import argparse
import pandas as pd
import pandasql as ps

seed(1)
app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

cmd_options = None
data_frame = None
events_list = [586, 606, 668, 693, 711, 804, 926, 1130, 1352, 1448, 1574, 1593, 1968, 1347, 1351]
events_ndx = 0
custom_round = True
first_false = True

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Project Wisconsin Trails</h1>'''

def getDictFromDf(df):
    # assumes one row in the df
    conv_response = {}
    for index, row in df.iterrows():
        conv_response['imagegroupid'] = row['image_group_id']
        conv_response['images'] = [row['image_url_1'], row['image_url_2'], row['image_url_3'], row['image_url_1_bbox'], row['image_url_2_bbox'], row['image_url_3_bbox']]
        conv_response['animalcount'] = row['count']
        conv_response['animaltype'] = row['species_name']
        conv_response['animaltype2'] = row['species_name']
        if (row['blank_image'] == True):
            conv_response['animaltype'] = "Blank"
            conv_response['animaltype2'] = "Blank"
        conv_response['event_id'] = row['event_id']
        animals = ["turkey", "rabbit", "fox", "bear", "coyote", "opossum", "raccoon", "deer", "elk", "wolf", "blank", "Blank"]
        if conv_response['animaltype'] not in animals:
            conv_response['animaltype'] = "Other"
    return conv_response

# A route to return new set of images.
@app.route('/api/v1/resources/newclassify', methods=['GET'])
def api_all():
    global cmd_options
    global data_frame
    global events_list
    global events_ndx
    global custom_round
    global first_false
    print(cmd_options)
    print(type(cmd_options))

    config_db = "database.ini"
    event_id = request.args.get('event_id', default=0, type=int)
    print("even_id in GET: ", event_id)     
    # query = "SELECT * FROM public.event_images where image_group_id='SSWI000000019636502'" #Deer
    # query = "SELECT * FROM public.event_images where image_group_id='SSWI000000017069780'"   #Elk
    if (event_id == 0):
        if (cmd_options.skip == 'custom'):
            event_id = events_list[0]
            events_ndx = 1
        else:
            event_id = random.randint(0, 2190)
    else:
        if (cmd_options.skip == '1'):
            event_id = event_id + 1
        elif (cmd_options.skip == 'random'):
            event_id = event_id + 1 + random.randint(1, 20)
        elif (cmd_options.skip == 'custom'):
            if (custom_round == True):
                event_id = events_list[events_ndx]
                events_ndx = events_ndx + 1
                if (events_ndx >= len(events_list)):
                    custom_round = False
            else:
                if (first_false == True):
                    event_id = random.randint(0, 2190)
                    first_false = False
                else:
                    event_id = event_id + 1 + random.randint(1, 20)
        if (event_id > 2190):
            event_id = random.randint(0, 2190)

    if (cmd_options.enable_db == 'false'):
        query = "SELECT * FROM data_frame where event_id=" + str(event_id)
        df = ps.sqldf(query)
    else:
        query = "SELECT * FROM public.event_images where event_id=" + str(event_id)
        df = load_db_table(config_db, query)
    print("print from the DB query run: ", df)
    conv_response = getDictFromDf(df)
    print("json conversion: ", conv_response)

    return jsonify(conv_response)
    # return jsonify(response_dict)

# FE posts JSON in this format: {'imagegroupid' :333, 'animal' :bobcat, 'animalcount : 4}
@app.route('/api/v1/resources/annotate', methods=['POST'])
def annotate():
    request_data = request.get_json()

    # Logic to update DB

    print("json : ", request_data)
    return jsonify("{'message' : 'success'}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_db', type=str, default='false', help='DB vs local CSV')
    parser.add_argument('--skip', type=str, default='random', help='random = event_id increments random(1..10); 1 = increment by 1; custom')
    opt = parser.parse_args()
    return opt

def main(cmd_opts):
    global cmd_options
    global data_frame
    cmd_options = cmd_opts
    print(cmd_options)
    data_frame = pd.read_csv("../../results/event_images_table.csv")
    app.run(debug=True,host='0.0.0.0')

if __name__ == "__main__":
    cmd_opts = parse_opt()
    main(cmd_opts)