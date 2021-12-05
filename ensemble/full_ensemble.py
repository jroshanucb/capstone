import pandas as pd
import numpy as np
from datetime import datetime
import json
import cv2
from pathlib import Path
import glob
import re
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
from matplotlib.pyplot import figure
from sklearn.metrics import classification_report

from ensemble.stage_1_ensemble import run_ensemble_stage_1
from ensemble.stage_2_ensemble import run_ensemble_stage_2
from ensemble.stage_3_ensemble import run_ensemble_stage_3

db_image_dir = 'https://wisconsintrails.s3.us-west-2.amazonaws.com/images/'
db_bbox_image_dir = 'https://wisconsintrails.s3.us-west-2.amazonaws.com/bboximages/'

def merge_ensemble_scripts(modelsz):
    #Merge stage 1 (blanks) and stage 2 (species)
    full_ensemble = pd.merge(run_ensemble_stage_1(modelsz), run_ensemble_stage_2(modelsz),
             on = 'image_group_id')

    #Stage 3: Counts and bboxes
    counts_bboxes = run_ensemble_stage_3(modelsz)


    #Merge All Stages to produce 2 tables (or csv)

    ##Full ensemble with counts (by event)
    full_ensemble = pd.merge(full_ensemble, counts_bboxes,
                        on = 'image_group_id', how = 'left')
    full_ensemble = full_ensemble.rename(columns={'ensemble_pred': 'blank',
                                 'event_final_pred': 'species'})
    full_ensemble['blank'] = full_ensemble['blank'].apply(lambda x: True if x == 'blank' else False)

    return full_ensemble

def full_ensemble_logic(full_ensemble, modelsz):
    '''

    Large Model
    ---Ensemble Logic---
    Blanks
    -If stage 1 returned blank:
        -Prediction = blank
    -If stage 1 returned species but stage 2 return blank:
        -Prediction = blank
    -If stage 1 and stage 2 returned species but count returned 0:
        -Prediction = blank

    species and counts
    -If stage 1 and stage 2 returned species and count > 0:
        -Prediction = species column (Determined in stage 2.py)
        -Count and bboxes = Yolo count unless MD count is less than Yolo count
                            OR count that is not 0
                            *Small and Medium models: Just yolo count

    '''
    group_id_list = []
    url_image_1 = []
    url_image_2 = []
    url_image_3 = []
    url_image_1_bbox = []
    url_image_2_bbox = []
    url_image_3_bbox = []
    blank_list = []
    final_pred_list = []
    final_top_3_list = []
    final_count_list = []
    final_bbox_list = []


    for row, values in full_ensemble.iterrows():
        #print(values['event_final_topk_conf'])

        final_top_3_list.append(values['event_final_topk_conf'])
        #Converting events back to image names
        img_grp = values['image_group_id']
        group_id_list.append(img_grp)

        if 'SSWI' in img_grp:
            image_appendices = ['A.jpg','B.jpg', 'C.jpg']
        else:
            image_appendices = ['_0A.jpeg','_1B.jpeg', '_2C.jpeg']

        #Raw images
        url_image_1.append(db_image_dir + img_grp + image_appendices[0])
        url_image_2.append(db_image_dir + img_grp + image_appendices[1])
        url_image_3.append(db_image_dir + img_grp + image_appendices[2])

        #Bbox images
        url_image_1_bbox.append(db_bbox_image_dir + img_grp + image_appendices[0])
        url_image_2_bbox.append(db_bbox_image_dir + img_grp + image_appendices[1])
        url_image_3_bbox.append(db_bbox_image_dir + img_grp + image_appendices[2])



        if (values['blank'] == True) or (values['species'] == 'blank') or\
        (values['yolo_count_max'] == 0 and values['md_count_max'] == 0):
            blank_list.append(True)
            final_pred_list.append('blank')
            final_count_list.append(0)
            final_bbox_list.append('None')
            final_top_3_list.append('')
        else:
            blank_list.append(False)
            final_pred_list.append(values['species'])

            if modelsz in ['small', 'medium']:
                final_count_list.append(values['yolo_count_max'])
                final_bbox_list.append(values['yolo_bbox'])

            elif modelsz == 'large':
                if values['yolo_count_max'] == 0:
                    final_count_list.append(values['md_count_max'])
                    final_bbox_list.append(values['md_bbox'])
                elif values['md_count_max'] == 0:
                    final_count_list.append(values['yolo_count_max'])
                    final_bbox_list.append(values['yolo_bbox'])

                elif (values['yolo_count_max'] <= values['md_count_max']):
                    final_count_list.append(values['yolo_count_max'])
                    final_bbox_list.append(values['yolo_bbox'])
                else:
                    final_count_list.append(values['md_count_max'])
                    final_bbox_list.append(values['md_bbox'])

    while("" in final_top_3_list) :
        final_top_3_list.remove('')

    event_images_table = pd.DataFrame({
    'image_group_id': group_id_list,
    'image_url_1':url_image_1,
    'image_url_2': url_image_2,
    'image_url_3': url_image_3,
     'image_url_1_bbox': url_image_1_bbox,
    'image_url_2_bbox': url_image_2_bbox,
     'image_url_3_bbox': url_image_3_bbox,
    'count': final_count_list,
    'species_name': final_pred_list,
    'blank_image': blank_list,
    'bboxes': final_bbox_list
    #'event_final_topk_conf': final_top_3_list
    })

    event_images_table.insert(0, 'event_id', range(0, len(event_images_table)))

    #Load date
    now = datetime.now() # current date and time
    date_string = now.strftime("%m/%d/%Y")
    event_images_table['load_date'] = date_string

    return event_images_table

def print_metrics(event_images_table, truth_file_path):
    #Test to Ground truth
    truth_file = pd.read_csv(truth_file_path)

    truth_pred_df = pd.merge(truth_file, event_images_table,
             right_on = 'image_group_id',
             left_on = 'TRIGGER_ID',
             how = 'left')
    truth_pred_df = truth_pred_df.drop_duplicates(subset = 'image_group_id')

    #Filter out Other and Blank
    truth_pred_df_no_other = truth_pred_df[truth_pred_df['CLASS_SPECIES_RESTATED'] != 'other']


    ###Confusion Matrix
    figure(figsize=(14, 10), dpi=80)
    y_test= truth_pred_df_no_other['CLASS_SPECIES_RESTATED']
    label_list = np.unique(y_test)
    pred= truth_pred_df_no_other['species_name']
    cm = confusion_matrix(y_test, pred, labels = label_list)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues"); #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel(
        'True labels');
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(label_list); ax.yaxis.set_ticklabels(label_list);
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 45)
    plt.show()

    ###Classification Report
    y_true = truth_pred_df_no_other['CLASS_SPECIES_RESTATED']
    y_pred = truth_pred_df_no_other['species_name']

    print(classification_report(y_true, y_pred, target_names=label_list))

def increment_path(path, exist_ok=False, sep='', mkdir=True):
    # Increment file or directory path, i.e. bbox_images/exp --> bbox_images/exp{sep}2, bbox_images/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def print_bbox_images(event_images_table, read_img_directory, write_img_directory,
                        img_size):

    for img1_path,img2_path,img3_path, bbox_json in zip(event_images_table['image_url_1_bbox'],
                              event_images_table['image_url_2_bbox'],
                              event_images_table['image_url_3_bbox'],
                              event_images_table['bboxes']):

        img1_name = img1_path.split('/')[-1]
        img2_name = img2_path.split('/')[-1]
        img3_name = img3_path.split('/')[-1]

        for img_name in [img1_name, img2_name, img3_name]:

            try:
                image = cv2.imread(read_img_directory + img_name)

                if bbox_json == 'None':
                    cv2.imwrite('../results/bbox_images/' + img_name, image)
                else:
                    if isinstance(bbox_json, dict):
                        bbox_dict = bbox_json
                    else:
                        bbox_dict = json.loads(bbox_json.replace("'", '"'))

                    if img_name == img1_name:
                        bbox_list = bbox_dict['A']
                    elif img_name == img2_name:
                        bbox_list = bbox_dict['B']
                    elif img_name == img3_name:
                        bbox_list = bbox_dict['C']

                    if bbox_list == []:
                        cv2.imwrite('bbox_images/' + img_name, image)
                        continue

                    if isinstance(bbox_list[0] , list):
                        bbox_to_write = bbox_list[0]
                    else:
                        bbox_to_write = bbox_list

                    for bbox_coords in bbox_to_write:

                        try:
                            x_center = float(bbox_coords[0].split(',')[0])*img_size
                            y_center = float(bbox_coords[0].split(',')[1])*img_size
                            width = float(bbox_coords[0].split(',')[2]) *img_size
                            height = float(bbox_coords[0].split(',')[3])*img_size
                        except:
                            x_center = float(bbox_coords.split(',')[0])*img_size
                            y_center = float(bbox_coords.split(',')[1])*img_size
                            width = float(bbox_coords.split(',')[2]) *img_size
                            height = float(bbox_coords.split(',')[3])*img_size

                        start_x = (x_center) - ((width)/2)
                        start_y = (y_center) - ((height)/2)
                        end_x = x_center + width
                        end_y = y_center + height

                        start_point = (int(start_x), int(start_y) )
                        end_point = (int(end_x), int(end_y) )

                        # Blue color in BGR
                        color = (255, 0, 0)

                        # Line thickness of 2 px
                        thickness = 2

                        # Using cv2.rectangle() method
                        # Draw a rectangle with blue line borders of thickness of 2 px
                        image = cv2.rectangle(image, start_point, end_point, color, thickness)

                    cv2.imwrite(write_img_directory + '/' + img_name, image)
            except:
                print("{} not written to bbox images.".format(img_name))
                continue

def run_full_ensemble(modelsz = 'small',
    read_img_directory = '/Users/sleung2/Documents/MIDS Program/Capstone_local/snapshot_wisconsin/all/yolo_splits4.2/test/images/',
    truth_file_path = '../data/test_labels.csv',
    write_images = 'true',
    img_size=329):
    full_ensemble = merge_ensemble_scripts(modelsz)
    full_ensemble.to_csv('../results/full_ensemble_int.csv', index = False)
    event_images_table = full_ensemble_logic(full_ensemble, modelsz)

    event_images_table.to_csv('../results/event_images_table.csv', index = False)

    if truth_file_path != None:
        print_metrics(event_images_table, truth_file_path)

    if write_images == 'true':
        write_img_directory = str(increment_path('../results/bbox_images/exp'))
        print_bbox_images(event_images_table, read_img_directory, write_img_directory,
                            img_size)

    return
