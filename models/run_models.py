import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

from run_yolo import run_format_yolo
from run_effnet import run_format_effnet
from run_megad import run_format_megad
import torch

def run_models(img_directory):
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

    #Model 1: Yolo Blank
    model_1_df = run_format_yolo(img_directory, 'yolov5l_best_blank.pt', stage_1_labels, 1,
                                                        run_blur = False)
    torch.cuda.empty_cache()

    #Model 2: Effnet Blank
    model_2_weights_path = 'efficientnetb0_50epochs_finetuned_model_yolosplits3_blanks.pt'
    model_2_df = run_format_effnet(img_directory, model_2_weights_path, stage_1_labels, 2)
    torch.cuda.empty_cache()

    #Model 3: Yolo Species
    model_3_df = run_format_yolo(img_directory, 'yolov5x_splits4_best.pt', stage_2_yolo_labels, 3,
                                                    run_blur = True)
    torch.cuda.empty_cache()

    #Model 4: Effnet Species
    model_4_weights_path = 'efficientnetb5_100epochs_finetuned_model_yolosplits4_BasePlusBlank.pt'
    model_4_df = run_format_effnet(img_directory, model_4_weights_path, stage_2_effnet_labels, 4)
    torch.cuda.empty_cache()

    #Model 5: Megadetector
    model_5_path = '../results/JSON_txt_outputs/phase2_megadetector_classifications_yolosplits_4-1_YOLO.json'
    model_5_df = run_format_megad(model_5_path, 5)
    #model_5_df.to_csv('model_5_df.csv', index = False)

    full_model_output = pd.concat([model_1_df,
                model_2_df,
                model_3_df,
                model_4_df,
                model_5_df])

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

    full_model_output.to_csv('full_model_output.csv', index = False)
    return

if __name__ == "__main__":
    img_directory = sys.argv[1]
    run_models(img_directory)
