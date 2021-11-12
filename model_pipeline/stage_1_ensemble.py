import os
import random
import shutil, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pylab as pl
from matplotlib.pyplot import figure
from collections import Counter

from os import listdir
from os.path import isfile, join

#Read lines from txt results file

top_path = 'phase 1-blank/'
yolo_file = 'phase1_yolo.txt'

effnet_file = 'phase1_efficientnetb0_classifications.json'

#HELPER FUNCTIONS

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

## YOLO
def yolo_blank_read_file(top_path, yolo_file):
    '''Convert Yolo txt file to predictions df by event'''

    with open(top_path + yolo_file, 'r') as f:
        blank_yolo_lines = f.readlines()

    yolo_image_prediction_list = []
    yolo_blank_prediction_list = []


    for line in blank_yolo_lines:

        image_name = line.split('Filename: ')[1].split('.jpg;')[0]
        prediction = line.split('.jpg;')[1].split('; Bbox[list]')[0]
        yolo_image_prediction_list.append(image_name)
        yolo_blank_prediction_list.append(prediction)

    yolo_stage_1_prediction_df = pd.DataFrame({'id': yolo_image_prediction_list, 'image_prediction': yolo_blank_prediction_list})

    #Create necessary columns
    yolo_stage_1_prediction_df['event_id'] = images_to_events(yolo_stage_1_prediction_df)
    yolo_stage_1_prediction_df['class_name'] = yolo_stage_1_prediction_df['prediction'].apply(lambda x: 'empty' if x == '' else 'animal')
    yolo_stage_1_prediction_df['class'] = yolo_stage_1_prediction_df['class_name'].apply(lambda x: 0 if x == 'animal' else 1)

    yolo_stage_1_prediction_grouped = group_events(yolo_stage_1_prediction_df)

    yolo_stage_1_prediction_grouped['event_prediction'] =  interpret_event_predictions(yolo_stage_1_prediction_grouped)

    return yolo_stage_1_prediction_grouped

## Efficientnet
def effnet_blank_read_file(top_path, yolo_file):
    '''Convert Effnet json file to predictions df by event'''

    effnet_stage_1 = pd.read_json(top_path + yolo_file)
    effnet_stage_1 = effnet_stage_1['phase1_classification_results'].apply(pd.Series)

    effnet_stage_1['event_id'] = images_to_events(image_df)

    effnet_stage_1_grouped = group_events(effnet_stage_1)

    effnet_stage_1_grouped['event_prediction'] =  interpret_event_predictions(effnet_stage_1_grouped)



    return effnet_stage_1_grouped
