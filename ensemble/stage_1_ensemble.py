import os
import random
import shutil, os
import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join

top_path = '/Users/sleung2/Documents/MIDS Program/capstone/ensemble/phase 1-blank/'
yolo_file = 'phase1_yolo.txt'
effnet_file = 'phase1_efficientnetb0_classifications.json'

#Threshold at which to overwrite effnet with yolo on empty images
conf_thresh = .75

def images_to_events(image_df, image_id_col = 'id'):
    '''Convert images to events'''

    return image_df['id'].apply(lambda x: x[:-5] if 'SSWI' in x else x.split('_')[0])

def group_events(prediction_image_df):
    '''Convert predictions in terms of events'''
    prediction_event_df = pd.pivot_table(prediction_image_df, index = 'event_id',
              columns = 'class_name',
              values = 'class',
              aggfunc=len, fill_value=0).reset_index() #len(x.unique())).fillna(0)
    return prediction_event_df

def interpret_event_predictions(prediction_event_df):
    return prediction_event_df.apply(lambda x : 'empty' if x['animal'] == 0 else 'animal', axis = 1)

def conf_score_calcs(stage_1, stage_1_pivot, model_type):
    animal_event_ids = stage_1_pivot[~(stage_1_pivot['event_prediction'] == 'empty')]['event_id']
    stage_1_animal = stage_1[stage_1['event_id'].isin(animal_event_ids)]
    stage_1_animal[stage_1_animal['class_name'] == 'animal']
    stage_1_animal_conf = stage_1_animal[['event_id','conf']].groupby('event_id').mean()

    stage_1_blank = stage_1[~stage_1['event_id'].isin(animal_event_ids)]
    stage_1_blank_conf = stage_1_blank[['event_id','conf']].groupby('event_id').mean()

    effnet_conf_scores = pd.concat([stage_1_animal_conf, stage_1_blank_conf])

    return effnet_conf_scores

## YOLO
def yolo_blank_read_file(top_path, yolo_file):
    '''Convert Yolo txt file to predictions df by event'''

    with open(top_path + yolo_file, 'r') as f:
        blank_yolo_lines = f.readlines()

    yolo_image_prediction_list = []
    yolo_blank_prediction_list = []


    for line in blank_yolo_lines:

        image_name = line.split('Filename: ')[1].split('; ')[0]
        prediction = line.split('; ')[1].split('; Bbox[list]')[0]

        yolo_image_prediction_list.append(image_name)
        yolo_blank_prediction_list.append(prediction)

    yolo_stage_1_prediction_df = pd.DataFrame({'id': yolo_image_prediction_list, 'image_prediction': yolo_blank_prediction_list})

    #Create necessary columns
    yolo_stage_1_prediction_df['event_id'] = images_to_events(yolo_stage_1_prediction_df)
    yolo_stage_1_prediction_df['class_name'] = yolo_stage_1_prediction_df['image_prediction'].apply(lambda x: 'empty' if x == '' else 'animal')
    yolo_stage_1_prediction_df['class'] = yolo_stage_1_prediction_df['class_name'].apply(lambda x: 0 if x == 'animal' else 1)

    yolo_stage_1_prediction_grouped = group_events(yolo_stage_1_prediction_df)

    yolo_stage_1_prediction_grouped['event_prediction'] =  interpret_event_predictions(yolo_stage_1_prediction_grouped)

    return yolo_stage_1_prediction_grouped

## Efficientnet
def effnet_blank_read_file(top_path, effnet_file):
    '''Convert Effnet json file to predictions df by event'''

    effnet_stage_1 = pd.read_json(top_path + effnet_file)
    effnet_stage_1 = effnet_stage_1['phase1_classification_results'].apply(pd.Series)

    effnet_stage_1['event_id'] = images_to_events(effnet_stage_1)

    effnet_stage_1_grouped = group_events(effnet_stage_1)

    effnet_stage_1_grouped['event_prediction'] =  interpret_event_predictions(effnet_stage_1_grouped)

    conf_scores = conf_score_calcs(effnet_stage_1, effnet_stage_1_grouped, 'effnet')

    effnet_stage_1_pred_conf = pd.merge(effnet_stage_1_grouped, conf_scores,
         how = 'left',
         left_on = 'event_id',
         right_index = True)[['event_id', 'event_prediction', 'conf']]

    return effnet_stage_1_pred_conf

def model_pred_merge(yolo_stage_1_pred_conf, effnet_stage_1_pred_conf):
    '''Merge yolo and effnet predictions into single df '''
    common = effnet_stage_1_pred_conf[effnet_stage_1_pred_conf['event_id'].isin(yolo_stage_1_pred_conf.event_id)]
    common_merged = pd.merge(common, yolo_stage_1_pred_conf,
             on = 'event_id',
             how = 'left')
    common_merged = common_merged.rename(columns = {'event_prediction_x': 'effnet_pred',
                                                   'event_prediction_y': 'yolo_pred'})
    common_merged = common_merged.drop(columns = ['animal', 'empty'])

    return common_merged

def ensemble_pred_logic(ensemble_row, conf_thresh):
    '''Function designed to run lambda, row by row to convert yolo and effnet_file
    preds into ensemble preds. Optimal performance seen when we only overwrite effnet_file
    with yolo on the empties and at a certain threshold'''

    if ensemble_row['effnet_pred'] == 'empty' and ensemble_row['conf'] < conf_thresh:
        ensemble_pred =  ensemble_row['yolo_pred']
    else:
        ensemble_pred = ensemble_row['effnet_pred']

    return ensemble_pred


def main():
    yolo_stage_1_pred_conf = yolo_blank_read_file(top_path, yolo_file)
    effnet_stage_1_pred_conf = effnet_blank_read_file(top_path, effnet_file)
    merged_stage_1_pred_conf = model_pred_merge(yolo_stage_1_pred_conf, effnet_stage_1_pred_conf)

    merged_stage_1_pred_conf['ensemble_pred'] = merged_stage_1_pred_conf.apply(lambda x: ensemble_pred_logic(x, conf_thresh), axis = 1)

    merged_stage_1_pred_conf.to_csv('/Users/sleung2/Documents/MIDS Program/capstone/ensemble/merged_stage_1_pred_conf.csv', index = False)

main()
