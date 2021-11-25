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

# Create some test data
response = {
    "imagegroupid": "111",
    "images": ["https://panoptes-uploads.zooniverse.org/subject_location/c9be3258-8b30-4e72-912a-a0185220bc30.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/f070cfa7-be52-43cb-87f6-da52aa31e33c.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/d6b4e341-16a9-45df-a6b5-101e8bfb822e.jpeg"],
    "animalcount": 2,
    "animaltype": "deer"
}

response2 = {
    "imagegroupid": "222",
    "images": ["https://panoptes-uploads.zooniverse.org/subject_location/f3fa28aa-a3fc-4b56-8274-e205fef4cb89.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/2d3e56bd-777f-4402-aa01-45bd92ad6e47.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/276a87b1-ad51-4149-8ffe-5ba60363a8a7.jpeg"],
    "animalcount": 9,
    "animaltype": "bear"
}

response3 = {
    "imagegroupid": "333",
    "images": ["https://panoptes-uploads.zooniverse.org/subject_location/60b9fcfa-20c0-45db-823a-4ce78ea1d740.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/ed1c603b-c741-44ad-ae11-3cdf404a3fb5.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/4804d5e2-3a2e-4ce6-b7c5-cc9397bb4a59.jpeg"],
    "animalcount": 4,
    "animaltype": "bobcat"
}

response4 = {
    "imagegroupid": "444",
    "images": ["https://panoptes-uploads.zooniverse.org/subject_location/337f00b0-5366-47f2-aaac-3f3c388cdae3.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/75e5a457-ee22-497f-921c-fa98a1c93963.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/63190ae8-e62a-4548-830f-a140e5899e70.jpeg"],
    "animalcount": 6,
    "animaltype": "turkey"
}

response5 = {
    "imagegroupid": "555",
    "images": ["https://panoptes-uploads.zooniverse.org/subject_location/2717792b-62ef-4063-b5b4-a6bcc398caa2.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/0104ae28-2c7b-4965-8ca5-331b7a2a2527.jpeg", "https://panoptes-uploads.zooniverse.org/subject_location/7aad0b27-7b66-4142-9f4d-74096f76c8f5.jpeg"],
    "animalcount": 7,
    "animaltype": "turkey"
}

response6 = { 
    'imagegroupid': 'SSWI000000019636502', 
    'images': ['https://capstone-trails-cam.s3.us-west-2.amazonaws.com/sswisimages/SSWI000000019636502C.jpg', 'https://capstone-trails-cam.s3.us-west-2.amazonaws.com/sswisimages/SSWI000000019636505A.jpg', 'https://capstone-trails-cam.s3.us-west-2.amazonaws.com/sswisimages/SSWI000000019636505B.jpg'], 
    'animalcount': 1, 
    'animaltype': 'deer'
}


sequence = [9, 4, 6, 1, 2, 3, 7]


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Project WI</h1>'''

def getDictFromDf(df):
    # assumes one row in the df
    conv_response = {}
    for index, row in df.iterrows():
        # conv_response = "{ 'imagegroupid': '" + row['image_group_id'] + "', 'images': ['" + row['image_url_1'] + "', '" 
        # conv_response += row['image_url_2'] + "', '" + row['image_url_3'] + "'" + "], " + "'animalcount': " + str(row['count']) + ", " 
        # conv_response += "'animaltype': '" + row['species_name'] + "'" + "}"
        conv_response['imagegroupid'] = row['image_group_id']
        conv_response['images'] = [row['image_url_1'], row['image_url_2'], row['image_url_3'], row['image_url_1_bbox'], row['image_url_2_bbox'], row['image_url_3_bbox']]
        conv_response['animalcount'] = row['count']
        conv_response['animaltype'] = row['species_name']
        conv_response['animaltype2'] = row['species_name']
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
    if (event_id == 0):
        event_id = random.randint(1, 26635)
    else:
        event_id = event_id + 1 + random.randint(1, 20)
        # event_id = event_id + 1
    # query = "SELECT * FROM public.event_images where image_group_id='SSWI000000017069780'"   #Elk
    query = "SELECT (event_id, image_group_id, image_url_1, image_url_2, image_url_3, image_url_1_bbox, image_url_2_bbox, image_url_3_bbox, count, species_name, load_date) FROM public.event_images where event_id=" + str(event_id)
    df = load_db_table(config_db, query)
    print("print from the DB query run: ", df)
    conv_response = getDictFromDf(df)
    print("json conversion: ", conv_response)
    # conv_response = response5
    # response_dict = json.loads(conv_response)
    

    return jsonify(conv_response)
    # return jsonify(response_dict)

# FE posts JSON in this format: {'imagegroupid' :333, 'animal' :bobcat, 'animalcount : 4}
@app.route('/api/v1/resources/annotate', methods=['POST'])
def annotate():
    request_data = request.get_json()

    # Logic to update DB

    print("json : ", request_data)
    return jsonify("{'message' : 'success'}")


app.run()
