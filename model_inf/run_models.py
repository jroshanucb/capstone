import argparse
import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from model_inf.run_yolo import run_format_yolo
from model_inf.run_effnet import run_format_effnet
from model_inf.run_megad import run_format_megad
import torch

def run_ensemble_models(img_directory,
                modelsz,
                dbwrite,
                imgsz):
    #images
    #img_directory = '/Users/sleung2/Documents/MIDS\ Program/Capstone_local/snapshot_wisconsin/all/yolo_splits4.2/test/images/'

    ##Labels
    #Stage 1

    stage_1_labels = pd.DataFrame(['animal', 'blank']).sort_values(0)
    stage_1_labels = stage_1_labels.rename(columns = {0: 'species'})
    stage_1_labels.insert(0, 'label', range(0, len(stage_1_labels)))

    #Stage 2 Yolo
    stage_2_yolo_labels = pd.DataFrame(['foxgray_foxred',
                  'cottontail_snowshoehare',
                  'raccoon',
                  'opossum',
                  'turkey',
                  'bear',
                  'elk',
                  'deer',
                  'coyote',
                  'wolf']).sort_values(0)
    stage_2_yolo_labels = stage_2_yolo_labels.rename(columns = {0: 'species'})
    stage_2_yolo_labels.insert(0, 'label', range(0, len(stage_2_yolo_labels)))

    #Stage 2 Effnet (Add blank)
    stage_2_effnet_labels = pd.DataFrame(['foxgray_foxred',
                  'cottontail_snowshoehare',
                  'raccoon',
                  'opossum',
                  'turkey',
                  'bear',
                  'elk',
                  'deer',
                  'coyote',
                  'wolf',
                  'blank']).sort_values(0)
    stage_2_effnet_labels = stage_2_effnet_labels.rename(columns = {0: 'species'})
    stage_2_effnet_labels.insert(0, 'label', range(0, len(stage_2_effnet_labels)))

    model_result_list = []

    print('''
    Stage 1: Running Blank Image Detection models
    ---------------------------------------------
    ''')
    if modelsz == 'small':
        total_model_num = 2
    elif modelsz == 'medium':
        total_model_num = 4
    elif modelsz == 'large':
        total_model_num = 5

    #Model 1: Yolo Blank
    if modelsz in ['medium', 'large']:


        print('''
        Running Yolov5 Blank, Model 1 of {}.
        '''.format(total_model_num))

        model_1_df = run_format_yolo(img_directory, 'weights/yolov5l_best_blank.pt', imgsz,
         stage_1_labels, 1, run_blur = False)
        model_result_list.append(model_1_df)

        torch.cuda.empty_cache()



    time.sleep(5)
    #Model 2: Effnet Blank
    if modelsz == 'small':
        print('''
        Running EfficientNet B0 Blank, Model 1 of {}.
        '''.format(total_model_num))
    elif modelsz in ['medium', 'large']:
        print('''
        Running EfficientNet B0 Blank, Model 2 of {}.
        '''.format(total_model_num))

    model_2_weights_path = 'weights/efficientnetb0_50epochs_finetuned_model_yolosplits3_blanks.pt'
    model_2_df = run_format_effnet(img_directory, model_2_weights_path, 224, stage_1_labels, 2)

    model_result_list.append(model_2_df)
    torch.cuda.empty_cache()

    time.sleep(5)
    #Model 3: Yolo Species
    if modelsz == 'small':
        print('''
        Running Yolov5 Species and Count, Model 2 of {}.
        '''.format(total_model_num))
    elif modelsz in ['medium', 'large']:
        print('''
        Running Yolov5 Species and Count, Model 3 of {}.
        '''.format(total_model_num))

    model_3_df = run_format_yolo(img_directory, 'weights/yolov5x_splits4_best.pt', imgsz,
     stage_2_yolo_labels, 3, run_blur = True)
    model_result_list.append(model_3_df)

    torch.cuda.empty_cache()

    if modelsz in ['medium', 'large']:
        time.sleep(5)
        #Model 4: Effnet Species
        print('''
        Running EfficientNet Species, Model 4 of {}.
        '''.format(total_model_num))

        model_4_weights_path = 'weights/efficientnetb5_25epochs_finetuned_model_yolosplits4_456_BasePlusBlank.pt'
        model_4_df = run_format_effnet(img_directory, model_4_weights_path, 456, stage_2_effnet_labels, 4)
        model_result_list.append(model_4_df)
        torch.cuda.empty_cache()

    if modelsz in ['large']:
        time.sleep(5)
        #Model 5: Megadetector
        print('''
        Running Megadetector Count, Model 5 of 5.
        ''')

        model_5_json_path = 'phase2_megadetector_output_yolosplits4-1.json'
        model_5_df = run_format_megad(img_directory, model_5_json_path, 5)
        model_result_list.append(model_5_df)

    full_model_output = pd.concat(model_result_list)


    #Load date
    now = datetime.now() # current date and time
    date_string = now.strftime("%m/%d/%Y")
    full_model_output['load_date'] = date_string

    #Reorder columns- model id to front
    cols = list(full_model_output)
    cols.insert(0, cols.pop(cols.index('model_id')))
    # use loc to reorder
    full_model_output = full_model_output.loc[:, cols]

    #Sequential Output IDs
    full_model_output.insert(0, 'Model_Output_ID', range(0, len(full_model_output)))

    full_model_output.to_csv('../results/full_model_output.csv', index = False)
    return
