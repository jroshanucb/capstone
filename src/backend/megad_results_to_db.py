"""Load json data from megadetector into DB (modeid = 5)

Usage:
    $ python3 path/to/megad_results_to_db.py --source path/to/img.jpg --output_json path/to/out.json --dbwrite='false' -modelid='5'
    example:
    python3 megad_results_to_db.py --source "../../../data/test/yolo_splits3/test/images" --output_json="../../../project/models/model_outputs/phase2_megadetector_output_YOLO.json" --dbwrite='false' --modelid='5'

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
    print('model output -------->>>>>>>', model_output)

    # 'SSWI000000012002832': {
    # 	'SSWI000000012002832A.jpg': {'Count': 1, 'Coords': '0.3131,0.6797,0.3067,0.369'}, 
    # 	'SSWI000000012002832C.jpg': {'Count': 1, 'Coords': '0.6738,0.2986,0.02073,0.02802'}, 
    # 	'SSWI000000012002832B.jpg': {'Count': 0, 'Coords': ''}
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
            if (value2['Count'] > 0): # for images where no species exist, coords will be empty
                image_id_count = value2['Count']
                image_id_coords = value2['Coords']
                image_id_conf = value2['Conf']
                # sql_values_stmt +=  "'" + image_id + "', '', '', " + str(image_id_count)
                sql_values_stmt +=  "'" + image_id + "', '', '" + image_id_conf + "', " + str(image_id_count)
                sql_values_stmt +=  ", false, false, '" + image_id_coords + "', "
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
    speciesList = ['bear', 'cottontail_snowshoehare', 'coyote', 'deer', 'elk', 'foxgray_foxred', 'opossum', 'raccoon', 'turkey', 'wolf']
    idx = int(id)
    if idx > 9 or idx < 0:
        speciesName = 'other'
    else:
        speciesName = speciesList[idx]
    return speciesName

def load_megad_json(output_json):
    megad_json = {}

    with open(output_json) as json_file:
        data = json.load(json_file)
    
    data = data['phase2_classification_results']
    
    # key, value in image_id is in the format: '0': "SSWI000000020143548C.jpg"
    for key, value in data.items():
        for dict_list in value:
            key = dict_list['img_id']
            megad_json[key] = {}
            megad_json[key]['Count'] = len(dict_list['detections'])
            coord_list = ""
            conf_list = ""
            for bbox_conf in dict_list['detections']:
                # coords has a list of 4 elements
                bbox = ','.join([str(item) for item in bbox_conf['bbox']]) + ";"
                coord_list += bbox
                conf_list += str(bbox_conf['conf']) + ";"
            if (len(dict_list['detections']) == 0):
                megad_json[key]['Coords'] = ""
                megad_json[key]['Conf'] = ""
            else:
                megad_json[key]['Coords'] = coord_list[:-1]
                megad_json[key]['Conf'] = conf_list[:-1]

    # print(megad_json)
    return megad_json

def getPreds(filename, megad_json):
    ret_preds = {}

    ret_preds = megad_json.get(filename, "empty")
    if ret_preds == "empty":
        ret_preds = {}

    return ret_preds

def process_images(
        source='test/images',  # path from where files have to be processed
        modelid='5',  # megadetector model
        output_json='../../../project/models/model_outputs/phase2_megadetector_output_YOLO.json', # json with output of mega detector run
        dbwrite='false', # flag to control write to DB
        ):
    global cmd_options

    iteration = 0
    # Organize events into a dictionary
    imagesDict = organize_events(source)
    numEvents = len(imagesDict.keys())

    megad_json = load_megad_json(output_json)
    if (dbwrite != 'false'):
        conn = db_init()

    # for every event from the event list, perform yolo inference for all images from an event
    model_output = {}
    count = 1
    for key, value in imagesDict.items():
        fileInfer = {}
        for id in value:
            filename = key + id

            # get data from megadetector json
            ret_preds= getPreds(filename, megad_json)

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
    parser.add_argument('--output_json', type=str, default='../../../project/models/model_outputs/phase2_megadetector_output_YOLO.json', help='path to json output')
    parser.add_argument('--modelid', type=str, default='4', help='5 = megadetector counts model')
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