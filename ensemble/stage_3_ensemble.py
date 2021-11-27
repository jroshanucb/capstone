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
ROOT = '../'

#YOLO Counts Logic
#
#Counts
# Events with multiple images of same class- we will take the max of the majority class
# Events with all different labels will get labelled with the count of highest conf score label

def load_ground_truth(foldername=os.path.join(ROOT,"data/") , filename="test_labels4-1.csv"):

    ground_truth = pd.read_csv(foldername + filename)

    return ground_truth

def load_megadetector_output(foldername="results/JSON_txt_outputs/", filename='phase2_megadetector_classifications_yolosplits_4-1_YOLO.json'):#filename="phase2_megadetector_output_YOLO.json"):
    """
    Pkg dependencies: os, glob, re, pandas
    Purpose:
    Inputs:
    Outputs:
    """

    with open(os.path.join(ROOT,foldername, filename), 'r') as fin:
        fobj = fin.read()
        megadetector = json.loads(fobj)

    event_list = []
    img_list = []
    detection_list = []

    for event, image_set in megadetector['phase2_classification_results'].items():
        for image in image_set:
            event_list.append(image['event_id'])
            img_list.append(image['img_id'])
            detection_list.append(image['detections'])

    megadetector_df = pd.DataFrame({'event_id': event_list,
                  'image_id':img_list,
                  'detections':detection_list})

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


    return megadetector_df

def split_and_convert(s):
    """
    Purpose: Utility function used in load_yolo_output function for bounding box.
    """
    new = []
    out = s.split(',')
    for i in out:
        new.append(round(float(i), 4))
    return new

def load_yolo_output(foldername="results/JSON_txt_outputs/", filename="phase2_yolo_yolosplits4_1.txt"):
    """
    Pkg dependencies: os, glob, re, pandas
    Purpose:
    Inputs:
    Outputs:

    """



    # Load yolo model output file
    with open(os.path.join(ROOT, foldername, filename), 'r') as fin:
        yolov5 = fin.readlines()

    # Parse through file and pick out filename and bounding box
    filenames = []
    bbox = []
    for line_num, line in enumerate(yolov5):
        newline = line.split("\n")[0]
        semicolon_idxs = [m.start() for m in re.finditer(";", newline)]
        bbox_start, bbox_end = re.search(r"Bbox\[list]:", newline).start(), re.search(r"Bbox\[list]:", newline).end()

        for i, idx in list(zip(range(0,len(semicolon_idxs)), semicolon_idxs)):
            # Filename
            if i == 0:
                filenames.append(newline[:idx].split("Filename: ")[1])#.lstrip()[:-4])

        # Yolo Bounding box
        bbox_data = newline[bbox_end:].lstrip().split(';')[:-1]
        if len(bbox_data) == 0:
            bbox.append([])
        else:
            subl = [split_and_convert(i) for i in bbox_data]
            bbox.append(subl)

    # Construct DataFrame
    yolov5 = pd.DataFrame([pd.Series(filenames), pd.Series(bbox)]).T
    yolov5.columns = ["image_id", "yolo_bbox"]
    yolov5.sort_values(by="image_id", inplace=True, ignore_index=True)
    yolov5['yolo_count'] = yolov5['yolo_bbox'].apply(lambda x: len(x))

    return yolov5

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

    final_counts['final_count'] = final_counts.apply(lambda x: x['yolo_count_max'] if x['yolo_count_max'] < x['md_count_max'] else x['md_count_max'], axis = 1)

    final_counts['final_bbox'] = final_counts.apply(lambda x: x['yolo_bbox'] if x['yolo_count_max'] < x['md_count_max'] else x['md_bbox'], axis = 1)

    final_counts = final_counts.rename(columns = {'event_id': 'image_group_id'})

    return final_counts[['image_group_id', 'image_id', 'final_count', 'final_bbox']]



def run_ensemble_stage_3():

    ground_truth = load_ground_truth()
    megadetector = load_megadetector_output()
    yolov5 = load_yolo_output()
    final_counts = merge_md_yolo(yolov5, megadetector)

    return final_counts
