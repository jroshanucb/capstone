import os
import random
import shutil, os
import pandas as pd
import numpy as np
from collections import Counter

from os import listdir
from os.path import isfile, join

#YOLO Species Logic
# Species
#
# Events with multiple images of same class will be labeled the majority class
#   For event to be labeled Blank, all images must be blank
# Events with all different labels will get labelled with highest confidence score
#
#Counts
# Events with multiple images of same class- we will take the max of the majority class
# Events with all different labels will get labelled with the count of highest conf score label

labels = pd.DataFrame(['foxgray_foxred',
              'cottontail_snowshoehare',
              'raccoon',
              'opossum',
              'turkey',
              'bear',
              'elk',
              'deer',
              'coyote',
              'wolf']).sort_values(0)
labels = labels.rename(columns = {0: 'species'})
labels.insert(0, 'class', range(0, len(labels)))


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
def yolo_stage_2_read_file(top_path, yolo_file):
    #Txt file to dataframe
    #We are taking the TOP confidence score species for each image as the species
    #Count is the count for all bounding boxes drawn in image

    with open(top_path + yolo_file, 'r') as f:
        yolo_lines = f.readlines()

    rows = []

    for item in yolo_lines:

        #Class number
        item_split_1 = item.split(' Bbox[list]')[0]
        class_conf_list = item_split_1.split(';')[1:-2]

        conf_score_max = 0

        i = 0
        for class_conf in class_conf_list:
            conf_score_int = float(class_conf.split(',Conf:')[1])


            if conf_score_int > conf_score_max:
                conf_score_max = conf_score_int
                max_index = i

            i += 1
        try:
            class_string = class_conf_list[max_index].split(',')[0]
            class_num = float(class_string.split(':')[1])
            class_num = int(class_num)
        except:
            class_num = 'Blank'

        #Counts
        count_split_1 = item.split(' Bbox[list]')[-1]
        count_split_2 = count_split_1.replace(';', ',')
        count_split_3 = count_split_2.split(',')
        count = int((len(count_split_3)-1)/4)

        #Filename
        file_split_1 = item.split('Filename: ')[1]
        file_name = file_split_1.split(';')[0].split('.jpg')[0]

        row = [file_name, class_num, count, conf_score_max]

        rows.append(row)

    yolo_stage_2_pred = pd.DataFrame(rows, columns = ['filename', 'class', 'count', 'conf_score_max'])

    return yolo_stage_2_pred

def yolo_stage_2_image_to_event_pred(yolo_stage_2_pred):
    #Event column- remove A,B,C appendix
    yolo_stage_2_pred['event_id'] = yolo_stage_2_pred['filename'].str[:-1]


    ## Events with muliple images of same class
    #Species determination
    #Groupby event, class
    event_class_group_count = pd.DataFrame(yolo_stage_2_pred.groupby(['event_id', 'class']).count()['filename']).reset_index().rename(columns = {'filename': 'count_rows'})

    #Filter by rows that have count greater than 1
    event_class_group_count_majority = event_class_group_count[event_class_group_count['count_rows'] > 1]

    #Remove blank events that are not a consensus for all 3 images
    event_class_group_count_majority = event_class_group_count_majority[~((event_class_group_count_majority['class'] == 'Blank') &
                                    (event_class_group_count_majority['count_rows'] < 3))]

    #Count determination
    event_class_group_majority_counts = pd.merge(yolo_stage_2_pred, event_class_group_count_majority,
             how = 'inner',
             on = ['event_id', 'class'])

    event_class_majority_counts = event_class_group_majority_counts[['event_id', 'count']].groupby(['event_id']).max().reset_index()

    event_max_conf_score = event_class_group_majority_counts[['event_id', 'conf_score_max']].groupby(['event_id']).max().reset_index()

    majority_df = pd.merge(event_class_majority_counts, event_class_group_count_majority,
             on = 'event_id',
             how = 'left')[['event_id', 'count','class']]

    majority_df = pd.merge(majority_df, event_max_conf_score,
             on = 'event_id',
             how = 'left')

    ## Events with all different labels
    predictions_single_count = yolo_stage_2_pred[~yolo_stage_2_pred['event_id'].isin(majority_df['event_id'])]

    event_list = []
    pred_list = []
    count_list = []
    conf_score_max_list = []

    previous_event = ''
    conf_score_group_max = 0
    pred_class = ''
    conf_score_group_max = 0

    predictions_single_count_grouped = predictions_single_count.groupby(['event_id', 'class',
                                                                         'conf_score_max'])

    for group_name, group in predictions_single_count_grouped:

        current_event = group_name[0]
        current_class = group['class'].iloc[0]
        current_conf_score = group['conf_score_max'].iloc[0]
        current_count = group['count'].iloc[0]

        #Check if we are looking at a new event
        if current_event != previous_event:
            conf_score_max_list.append(conf_score_group_max)
            conf_score_group_max = 0
            event_list.append(previous_event)
            pred_list.append(pred_class)
            count_list.append(current_count)

        if conf_score_group_max < current_conf_score:
            pred_class = current_class
            conf_score_group_max = current_conf_score

        previous_event = current_event

    conf_score_max_list.append(conf_score_group_max)
    event_list.append(previous_event)
    pred_list.append(pred_class)
    count_list.append(current_count)

    single_class_df = pd.DataFrame(list(zip(event_list,
                  pred_list,
                  count_list,
                  conf_score_max_list)),
                columns = ['event_id', 'class', 'count','conf_score_max'])

    predictions_by_events_df = pd.concat([majority_df, single_class_df])


    yolo_preds_df = pd.merge(predictions_by_events_df, labels,
             how = 'left',
             on = 'class')

    yolo_preds_df = yolo_preds_df.rename(columns = {'class': 'pred_class',
                                                   'conf_score_max': 'conf_score',
                                                    'count': 'pred_count',
                                                   'species': 'pred_class_name'})
    return yolo_preds_df

## Efficientnet
def effnet_stage_2_read_file(top_path, effnet_file):
    '''Convert Effnet json file to predictions df by event'''

    model_results = pd.read_json(top_path + effnet_file)
    df = model_results['phase2_classification_results'].apply(pd.Series)
    df['event_id'] = df['id'].str.split('.jpg').str[0]
    df['event_id'] = df['event_id'].str[:-1]


    preds_dict = {}
    for index, row in df.iterrows():
        event_id = str(row['event_id'])
        pred_class = row['class']
        pred_conf = row['conf']
        #print(event_id, pred_class, pred_conf)

        result_dict = {
            "class": pred_class,
            "conf": pred_conf
        }

        if event_id in preds_dict:
            preds_dict[event_id].append(result_dict)
        else:
            preds_dict[event_id] = [result_dict]

    final_preds_dict = {}
    for key, value in preds_dict.items():
        event_id = key
        counts = Counter(d['class'] for d in value)

        ## if all 3 predictions are different, defer to class with highest confidence
        if len(counts) == 3:
            highest_conf = max([x['conf'] for x in value])
            pred_class = [x['class'] for x in value if x['conf']==highest_conf][0]

      ## if there is an even number of predictions (2), defer to class with higher confidence
        elif sum(counts.values()) < 3:
            highest_conf = max([x['conf'] for x in value])
            pred_class = [x['class'] for x in value if x['conf']==highest_conf][0]

      ## otherwise, class is based on majority class, conf score is based on highest score for
      ## majority class
        else:
            most_common = {'most_common': counts.most_common(1)[0][0]}
            pred_class = most_common['most_common']
            highest_conf = max([x['conf'] for x in value if x['class'] == most_common['most_common']])

        final_preds_dict[event_id] = [pred_class, highest_conf]

    effnet_stage_2_preds_df = pd.DataFrame.from_dict(final_preds_dict,  orient='index').reset_index()
    effnet_stage_2_preds_df.columns=['event_id', 'pred_class', 'conf_score']
    label_mapping = dict({0:'bear', 1:'cottontail_snowshoehare', 2:'coyote', 3:'deer', 4:'elk', 5:'foxgray_foxred', 6:'opossum', 7:'raccoon', 8:'turkey', 9:'wolf'})
    effnet_stage_2_preds_df['pred_class_name'] = effnet_stage_2_preds_df['pred_class'].map(label_mapping)

    return effnet_stage_2_preds_df

def model_pred_merge(yolo_stage_1_pred_conf, effnet_stage_1_pred_conf):
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
    preds into ensemble preds'''

    if ensemble_row['effnet_pred'] == 'empty' and ensemble_row['conf'] < conf_thresh:
        ensemble_pred =  ensemble_row['yolo_pred']
    else:
        ensemble_pred = ensemble_row['effnet_pred']

    return ensemble_pred



top_path = '/Users/sleung2/Documents/MIDS Program/capstone/ensemble/phase 2-species/'
yolo_file = 'output_no_wolf.txt'
effnet_file = 'phase2_efficientnetb5_classifications.json'
conf_thresh = .75

yolo_stage_2_pred = yolo_stage_2_read_file(top_path, yolo_file)
yolo_preds_df = yolo_stage_2_image_to_event_pred(yolo_stage_2_pred)

effnet_stage_2_preds_df = effnet_stage_2_read_file(top_path, effnet_file)
