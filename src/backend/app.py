from typing import Sequence
import flask
from flask import request, jsonify
from flask_cors import CORS
from random import seed
from random import choice
from db_conn import load_db_table
import json
import random
seed(1)
app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True


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
        animals = ["Turkey", "Cottontail", "Fox, Gray", "Fox, Red", "Bear", "Coyote", "Opossum", "Raccoon", "Snowshoe Hare", "Deer", "Elk", "Wolf"]
        if conv_response['animaltype'] not in animals:
            conv_response['animaltype'] = "Other"
    return conv_response

# A route to return new set of images.
@app.route('/api/v1/resources/newclassify', methods=['GET'])
def api_all():
    config_db = "database.ini"
    event_id = request.args.get('event_id', default=0, type=int)
    print("even_id in GET: ", event_id)
    # query = "SELECT * FROM public.event_images where image_group_id='SSWI000000019636502'" #Deer
    # query = "SELECT * FROM public.event_images where image_group_id='SSWI000000017069780'"   #Elk
    if (event_id == 0):
        event_id = random.randint(0, 2190)
    else:
        event_id = event_id + 1 + random.randint(1, 20)
        # event_id = event_id + 1
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


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
