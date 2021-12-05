import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob
import json
import re
import cv2
from PIL import Image

#STAGE 3- COUNTS

#Read lines from csv output file
ROOT = ''
foldername="results/"
filename='full_model_output.csv'

#YOLO Counts Logic
#
#Counts
# Events with multiple images of same class- we will take the max of the majority class
# Events with all different labels will get labelled with the count of highest conf score label

def load_megadetector_output(foldername="src/", filename='full_model_output.csv'):#filename="phase2_megadetector_output_YOLO.json"):
    """
    Pkg dependencies: os, glob, re, pandas
    Purpose:
    Inputs:
    Outputs:
    """

    output_file = pd.read_csv(os.path.join(ROOT,foldername, filename))
    megadetector = output_file[output_file['model_id'] == 5]

    image_group_ids = []
    image_ids = []
    detection_conf_list = []

    for row, value in megadetector.iterrows():
        for image in range(1,4):
            image_group_ids.append(value['image_group_id'])
            image_ids.append(value['image_id_{}'.format(image)])

            #BBOXES
            detection_int_string = value['image_id_{}_bbox'.format(image)]
            if isinstance(detection_int_string, float):
                detection_int_list = []
            else:
                detection_int_list = detection_int_string.split(';')
            #detection_list.append(detection_int_list)

            #CONF
            conf_int_string = value['image_id_{}_conf'.format(image)]
            if isinstance(conf_int_string, float):
                conf_int_list = []
            else:
                conf_int_list = conf_int_string.split(';')
            #conf_list.append(conf_int_list)

            #BBOXES and CONF to list of dicts
            detection_conf_int_list = []

            if len(detection_int_list) > 0:

                for bbox,conf in zip(detection_int_list, conf_int_list):
                    detection_conf_dict = {'bbox': bbox,
                                          'conf': conf}
                    detection_conf_int_list.append(detection_conf_dict)

            detection_conf_list.append(detection_conf_int_list)



    megadetector_df = pd.DataFrame({'event_id': image_group_ids,
                 'image_id': image_ids,
                 'detections':detection_conf_list})

    #Remove rows where there was no image for the event
    megadetector_df = megadetector_df[~megadetector_df['image_id'].isnull()]

    def extract_yolo(list_of_detections):
        yolo_list = []

        for i in list_of_detections:
            yolo_list.append(i['bbox'])
        return yolo_list

    megadetector_df['yolo'] = megadetector_df['detections'].apply(lambda x: extract_yolo(x))
    megadetector_df['count'] = megadetector_df['yolo'].apply(lambda x: len(x))

    def extract_conf(list_of_detections):
        conf_list = []

        for i in list_of_detections:
            conf_list.append(i['conf'])
        return conf_list

    megadetector_df['all_conf'] = megadetector_df['detections'].apply(lambda x: extract_conf(x))
    megadetector_df['max_detection_conf'] = megadetector_df['all_conf'].apply(lambda x:  max(x) if len(x) > 0 else 0)
    megadetector_df['all_class_pred'] = megadetector_df['count'].apply(lambda x:[1]*x)
    megadetector_df.loc[:, "length"] = megadetector_df['image_id'].apply(lambda x: len(x))
    megadetector_df['image_id'] = megadetector_df['event_id'] + megadetector_df['image_id']

    megadetector_df.drop(columns=['length'], inplace=True)

    return megadetector_df

def load_yolo_output(foldername="src/", filename='model_output_11202021_4.csv'):
    """
    Pkg dependencies: os, glob, re, pandas
    Purpose:
    Inputs:
    Outputs:

    """


    output_file = pd.read_csv(os.path.join(ROOT,foldername, filename))
    yolo = output_file[output_file['model_id'] == 3]

    image_group_ids = []
    image_ids = []
    detection_list = []

    for row, value in yolo.iterrows():
        for image in range(1,4):
            image_group_ids.append(value['image_group_id'])
            image_ids.append(value['image_id_{}'.format(image)])

            #BBOXES
            detection_int_list = []

            detection_int_string = value['image_id_{}_bbox'.format(image)]
            if isinstance(detection_int_string, float):
                detection_list.append(detection_int_list)
            else:
                detection_split_list = detection_int_string.split(';')
                for bbox in detection_split_list:

                    detection_int_list.append(detection_split_list)

                detection_list.append(detection_int_list)



    yolov5 = pd.DataFrame({'event_id': image_group_ids,
                 'image_id': image_ids,
                 'yolo_bbox':detection_list})

    yolov5['yolo_count'] = yolov5['yolo_bbox'].apply(lambda x: len(x))
    yolov5['image_id'] = yolov5['event_id'] + yolov5['image_id']

    return yolov5

+def non_merge_yolo_formatting(yolo_df):
    """
    Formatting yolo correctly when running small or medium size ensemble.
    """
    yolo_eventid_counts = yolo_df[['event_id', 'yolo_count']].groupby(by='event_id').agg('max')
    yolo_eventid_counts = yolo_eventid_counts.rename(columns = {'yolo_count': 'yolo_count_max'})
    final_counts = pd.merge(yolo_df,yolo_eventid_counts,
            how = 'left',
            on = 'event_id')

    final_counts['image_id_appendix'] = final_counts['image_id'].str[-1]

    event_id_group = []
    yolo_count_max_group = []
    yolo_bbox_group = []

    for group, values in final_counts.groupby(['event_id', 'yolo_count_max']):

        event_id_group.append(group[0])
        yolo_count_max_group.append(group[1])

        md_bbox_dict = {}
        yolo_bbox_dict = {}

        for image, yolo in zip(list(values['image_id_appendix']),  list(values['yolo_bbox'])):
            yolo_bbox_dict[image] = yolo

        yolo_bbox_group.append(yolo_bbox_dict)

    final_counts_bboxes = pd.DataFrame({'event_id': event_id_group,
                  'yolo_count_max': yolo_count_max_group,
                 'yolo_bbox': yolo_bbox_group})
    final_counts_bboxes = final_counts_bboxes.rename(columns = {'event_id': 'image_group_id'})

    final_counts_bboxes['md_count_max'] = ''
    final_counts_bboxes['md_bbox'] = ''

    return final_counts_bboxes

def merge_md_yolo(yolo_df, megadetector_df):
    """
    Pkg dependencies: pandas
    Purpose:
    Inputs: YOLO pd.DataFrame, Megadetector pd.DataFrame, ground truth pd.DataFrame
    Outputs: Merged pd.DataFrame of YOLO, Megadetector and ground truth
    """

    # Merge megadetector to YOLO by "image_id"
    megadetector_df = megadetector_df.rename(columns = {'yolo': 'md_bbox',
                                                   'count': 'md_count'})

    merged_raw = megadetector_df[['event_id','image_id','md_bbox', 'md_count']].merge(yolo_df[['image_id', 'yolo_bbox', 'yolo_count']], left_on="image_id", right_on="image_id")

    # Group by imageid (there should be 3), take the max count across the imageid that compose the event
    gby_eventid_counts = merged_raw[['event_id', 'md_count', 'yolo_count']].groupby(by='event_id').agg('max')
    gby_eventid_counts = gby_eventid_counts.rename(columns = {'md_count': 'md_count_max',
                                    'yolo_count': 'yolo_count_max'})

    final_counts = pd.merge(merged_raw, gby_eventid_counts,
         on = 'event_id', how = 'left')
    final_counts['image_id_appendix'] = final_counts['image_id'].str[-1]


    event_id_group = []
    md_count_max_group = []
    yolo_count_max_group = []

    md_bbox_group = []
    yolo_bbox_group = []

    for group, values in final_counts.groupby(['event_id', 'md_count_max', 'yolo_count_max']):

        event_id_group.append(group[0])
        md_count_max_group.append(group[1])
        yolo_count_max_group.append(group[2])

        md_bbox_dict = {}
        yolo_bbox_dict = {}

        for image, md, yolo in zip(list(values['image_id_appendix']), list(values['md_bbox']), list(values['yolo_bbox'])):
            md_bbox_dict[image] = md
            yolo_bbox_dict[image] = yolo


        md_bbox_group.append(md_bbox_dict)
        yolo_bbox_group.append(yolo_bbox_dict)

    final_counts_bboxes = pd.DataFrame({'event_id': event_id_group,
                   'md_count_max': md_count_max_group,
                  'yolo_count_max': yolo_count_max_group,
                  'md_bbox': md_bbox_group,
                 'yolo_bbox': yolo_bbox_group})
    final_counts_bboxes = final_counts_bboxes.rename(columns = {'event_id': 'image_group_id'})
    return final_counts_bboxes



def run_ensemble_stage_3(modelsz):

    yolov5 = load_yolo_output(foldername, filename)

    #If small or medium model, only run yolo
    if modelsz in ['small', 'medium']:
        final_counts_bboxes = non_merge_yolo_formatting(yolov5)

        return final_counts_bboxes[['image_group_id', 'md_count_max', 'yolo_count_max', 'md_bbox', 'yolo_bbox']]

    #If large model, only run yolo
    if modelsz == 'large':
        megadetector = load_megadetector_output(foldername, filename)
        final_counts_bboxes = merge_md_yolo(yolov5, megadetector)

        return final_counts_bboxes
