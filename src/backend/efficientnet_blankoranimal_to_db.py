"""Load json data from efficientnet into DB

Usage:
    $ python3 path/to/efficientnet_blankoranimal_to_db.py --source path/to/img.jpg --output_json path/to/out.json --dbwrite='false' -modelid='4'
    example:
    python3 efficientnet_blankoranimal_to_db.py --source "../../../data/test/yolo_splits3/test/images" --output_json "../../../project/models/model_outputs/phase1_efficientnetb0_classifications.json" --dbwrite='false' --modelid='4'

Author:
    Javed Roshan
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import os
import json

from db_conn import load_db_table
from db_conn import config
import pandas as pd
import psycopg2

cmd_options = None  # command options to be used during inference

# Organize all files into a dictionary of events. This assumes filesnames have "eventId + fileId.jpg" structure
def organize_events(
        source='test/images/'
    ):
    # populate the dictionary to check which events have images
    imagesDict = {}
    # there are 2 file formats: SSWI000000012915863A.jpg & 3004659_2C.jpeg
    count = 0
    for filename in os.listdir(source):
        filename_tokens = filename.strip().split('.')
        image_name = filename_tokens[0]
        if (image_name[0:2] == "SS"):
            eventId = image_name[:-1]
            imageId = image_name[-1:] + "." + filename_tokens[1]
        else:
            eventId = image_name[:-3]
            imageId = image_name[-3:] + "." + filename_tokens[1]
        count = count + 1
        dict_eventId = imagesDict.get(eventId, "empty")
        if (dict_eventId == "empty"):
            imagesDict[eventId] = [imageId]
        else:
            imagesDict[eventId] = imagesDict[eventId] + [imageId]

    return imagesDict

def get_insert_stmt():
    sql_stmt = "insert into public.model_output ("
    sql_stmt += "model_output_id, model_id, image_group_id, "
    sql_stmt += "image_id_1, image_id_1_species_name, image_id_1_conf, image_id_1_count, image_id_1_blank, image_id_1_detectable, image_id_1_bbox, " 
    sql_stmt += "image_id_2, image_id_2_species_name, image_id_2_conf, image_id_2_count, image_id_2_blank, image_id_2_detectable, image_id_2_bbox, "
    sql_stmt += "image_id_3, image_id_3_species_name, image_id_3_conf, image_id_3_count, image_id_3_blank, image_id_3_detectable, image_id_3_bbox, "
    sql_stmt += "load_date) values "
    return sql_stmt

def get_values_stmt(iteration, iter_size, modelid, model_output, numEvents):
    sql_values_stmt = ""

    # Example model_output[key] = 
    # {'SSWI000000020365431C.jpg': {'Class': ['8.0', '8.0'], 
    #                                'Conf': ['0.7531381249427795', '0.8462810516357422'], 
    #                                'Coords': ['0.9179331064224243,0.7872340679168701,0.12158054858446121,0.09118541330099106', '0.38145896792411804,0.9088146090507507,0.2705167233943939,0.18237082660198212']
    #                               }, 
    #  'SSWI000000020365431B.jpg': {'Class': ['8.0', '8.0'], 
    #                                'Conf': ['0.7363986968994141', '0.7579455971717834'], 
    #                                'Coords': ['0.34802430868148804,0.9103343486785889,0.22796352207660675,0.17933130264282227', '0.8875380158424377,0.7857142686843872,0.13981762528419495,0.09422492235898972']
    #                               },
    #  'SSWI000000020365431A.jpg': {'Class': ['8.0', '8.0', '5.0'], 
    #                                'Conf': ['0.337464839220047', '0.41901835799217224', '0.4265201687812805'], 
    #                                'Coords': ['0.7644376754760742,0.7963525652885437,0.12462005764245987,0.08510638028383255', '0.1322188377380371,0.9194529056549072,0.2644376754760742,0.16109421849250793', '0.1322188377380371,0.9179331064224243,0.2583586573600769,0.16413374245166779']
    #                               }
    # }
    found = False
    counter = 1
    model_num = int(modelid)
    for key, value in model_output.items():
        # 3611 / 3 = 1204 events total; add some buffer between model outputs i.e., 1250 events
        # yolo splits 4.1 has 2054 events; with 4961 images; add a buffer of 40 events across modelids
        model_output_id = (model_num-1) * (numEvents + 40) + iteration * iter_size + counter
        counter = counter + 1
        image_group_id = key # this is the event_id

        sql_values_stmt += "(" + str(model_output_id) + ", " + str(model_num) + ", '" + image_group_id + "', "
        for key2, value2 in value.items():
            dict1 = value2
            image_id = key2.strip().split('.')[0][-1:] # get 'C' from this file name 'SSWI000000020365431C.jpg' or '2008329_0A.jpeg'
            if (len(dict1.keys()) > 0): # for images where no species exist, the dict will be empty
                # ignore value2 for now
                # image_id_species_name = [get_speciesname_from_id(int(float(sn))) for sn in dict1['Class']]
                image_id_species_name = dict1['Class']
                image_id_conf = dict1['Conf']
                image_id_count = 0
                sql_values_stmt +=  "'" + image_id + "', '" + str(image_id_species_name) + "', '" + str(image_id_conf) + "', " + str(image_id_count)
                if (image_id_species_name == 'blank'):
                    sql_values_stmt +=  ", true, false, '', " # blank is true
                else:
                    sql_values_stmt +=  ", false, false, '', " # else it is animal, blank = false
            else:
                # empty image with no predictions
                sql_values_stmt +=  "'" + image_id + "', '', '', 0"
                sql_values_stmt +=  ", true, false, '', "

        event_size = len(value.keys())
        # if event has only 2 items, append nulls for the 3rd image
        if (event_size == 2):
            sql_values_stmt += "'', '', '', 0, false, false, '', "

        # if event has only 1 item, append nulls for the 2nd and 3rd image
        if (event_size == 1):
            sql_values_stmt += "'', '', '', 0, false, false, '', "
            sql_values_stmt += "'', '', '', 0, false, false, '', "

        load_date = "to_date('20-11-2021','DD-MM-YYYY')"
        sql_values_stmt += load_date + "), "

    return sql_values_stmt


def db_init():
    config_db = "database.ini"
    params = config(config_db)
    conn = psycopg2.connect(**params)

    return conn

def db_flush(iteration, iter_size, modelid, conn, model_output, numEvents):
    # model_output has the format of 
    # model_output[image_group_id] = dict of fileInfer
    # fileInfer has the format of 
    # fileInfer[filename] = image_id, class (a number), coordinates (count from these numbers)

    sql_insert_stmt = get_insert_stmt()
    sql_values_stmt = get_values_stmt(iteration, iter_size, modelid, model_output, numEvents)
    sql_stmt = sql_insert_stmt + sql_values_stmt[:-2]
    print("sql statment ---->", sql_stmt)
    cur = conn.cursor()
    cur.execute(sql_stmt)
    conn.commit()

    return

def get_speciesname_from_id(id):
    # 0 is animal and 1 is blank
    speciesList = ['animal', 'blank']
    idx = int(id)
    if idx > 9 or idx < 0:
        speciesName = 'other'
    else:
        speciesName = speciesList[idx]
    return speciesName

def load_effnet_json(output_json):
    effnet_json = {}

    with open(output_json) as json_file:
        data = json.load(json_file)
    
    data = data['phase1_classification_results']
    for dict_list in data:
        value = dict_list
        newKey = value['id']
        effnet_json[newKey] = {}

        effnet_json[newKey]['Class'] = value['class_name']
        effnet_json[newKey]['Conf'] = value['conf']

    # print(effnet_json)
    return effnet_json

def getPreds(filename, effnet_json):
    ret_preds = {}

    ret_preds = effnet_json.get(filename, "empty")
    if ret_preds == "empty":
        ret_preds = {}

    return ret_preds

def process_images(
        source='test/images',  # path from where files have to be processed
        modelid='4',  # 2 = efficientnet blank model; 4 = efficientnet species model
        output_json='../../../project/models/model_outputs/phase2_efficientnetb5_classifications.json', # json with output of efficientnet run
        dbwrite='false', # flag to control write to DB
        ):
    global cmd_options

    iteration = 0
    # Organize events into a dictionary
    imagesDict = organize_events(source)
    numEvents = len(imagesDict.keys())

    effnet_json = load_effnet_json(output_json)
    if (dbwrite != 'false'):
        conn = db_init()

    # for every event from the event list, perform yolo inference for all images from an event
    model_output = {}
    count = 1
    for key, value in imagesDict.items():
        fileInfer = {}
        for id in value:
            filename = key + id

            # get data from efficientnet json
            ret_preds= getPreds(filename, effnet_json)

            fileInfer[filename] = ret_preds
        model_output[key] = fileInfer
        # print(fileInfer)
        count += 1

        # db flush for every 50 events
        if (dbwrite=='true' and count > 50):
            db_flush(iteration, 50, modelid, conn, model_output, numEvents)
            iteration = iteration + 1
            model_output = {}
            count = 1
        elif count > 50:
            model_output = {}
            count = 1
    
    # final flush
    if (dbwrite=='true'):
        db_flush(iteration, 50, modelid, conn, model_output, numEvents)

    return

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='test/images/', help='path to get images for inference')
    parser.add_argument('--output_json', type=str, default='../../../project/models/model_outputs/phase2_efficientnetb5_classifications.json', help='path to json output')
    parser.add_argument('--modelid', type=str, default='4', help='2 = efficientnet blank model; 4 = efficientnet species model')
    parser.add_argument('--dbwrite', type=str, default='false', help='db persistence enabler')
    opt = parser.parse_args()
    return opt

def main(cmd_opts):
    global cmd_options
    cmd_options = cmd_opts
    process_images(**vars(cmd_options))

if __name__ == "__main__":
    cmd_opts = parse_opt()
    main(cmd_opts)