import os
import random
import shutil, os
import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join


#Read lines from txt results file
top_path = '../src/'
output_file = 'model_output_11202021_4.csv'

#HELPER FUNCTIONS

def event_prediction_for_model(row):
        if row['image_id_1_blank'] == True and row['image_id_2_blank'] == True and row['image_id_3_blank'] == True:
            return 'blank'
        else:
            return 'animal'

def event_conf_for_model(row):
    event_classification = row['event_prediction']
    if event_classification == 'blank':
        event_blank_bool = True
    else:
        event_blank_bool = False

    conf_score_list = []
    for image_num in range(1,4):
        if row['image_id_{}_blank'.format(image_num)] == event_blank_bool:
            conf_score_list.append(float(row['image_id_{}_conf'.format(image_num)]))

    return sum(conf_score_list)/ len(conf_score_list)


## YOLO
def blank_model_event_preds(top_path, output_file, model_id):
    output_df = pd.read_csv(top_path+ output_file)

    model_output_df = output_df[output_df['model_id'] == model_id]

    model_output_df['event_prediction'] = model_output_df.apply(lambda row: event_prediction_for_model(row), axis = 1)

    model_output_df['event_conf'] = model_output_df.apply(lambda row: event_conf_for_model(row), axis = 1)

    return model_output_df[[ 'image_group_id',
       'event_prediction', 'event_conf']]

def model_pred_merge(yolo_model_output, effnet_model_output):
    '''Merge yolo and effnet predictions into single df '''
    blank_model_output = pd.concat([yolo_model_output, effnet_model_output])

    blank_model_output_merged = pd.merge(effnet_model_output, yolo_model_output,
            on = 'image_group_id')

    blank_model_output_merged = blank_model_output_merged.rename(columns = {'event_prediction_x': 'effnet_pred',
                                                                            'event_conf_x':'effnet_conf',
                                                       'event_prediction_y': 'yolo_pred',
                                                                           'event_conf_y':'yolo_conf'})

    return blank_model_output_merged

def ensemble_pred_logic(ensemble_row, conf_thresh):
    '''Function designed to run lambda, row by row to convert yolo and effnet_file
    preds into ensemble preds. Optimal performance seen when we only overwrite effnet_file
    with yolo on the empties and at a certain threshold'''

    if ensemble_row['effnet_pred'] == 'blank' and ensemble_row['effnet_conf'] < conf_thresh:
        ensemble_pred =  ensemble_row['yolo_pred']
    else:
        ensemble_pred = ensemble_row['effnet_pred']

    return ensemble_pred


def run_ensemble_stage_1(conf_thresh = 1):
    '''conf_thresh: Threshold at which to overwrite effnet with yolo on empty images'''
    yolo_model_output = blank_model_event_preds(top_path, output_file, 1)
    effnet_model_output = blank_model_event_preds(top_path, output_file, 2)
    blank_model_output_merged = model_pred_merge(yolo_model_output, effnet_model_output)

    blank_model_output_merged['ensemble_pred'] = blank_model_output_merged.apply(lambda x: ensemble_pred_logic(x, conf_thresh), axis = 1)

    #blank_model_output_merged.to_csv('../ensemble/merged_stage_1_pred_conf.csv', index = False)

    return blank_model_output_merged[['image_group_id', 'ensemble_pred']]
