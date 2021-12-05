import pandas as pd
import numpy as np
from datetime import datetime
import json

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pylab as pl
from matplotlib.pyplot import figure
from sklearn.metrics import classification_report

from stage_1_ensemble import run_ensemble_stage_1
from stage_2_ensemble import run_ensemble_stage_2
from stage_3_ensemble import run_ensemble_stage_3

image_dir = 'https://wisconsintrails.s3.us-west-2.amazonaws.com/images/'
bbox_image_dir = 'https://wisconsintrails.s3.us-west-2.amazonaws.com/bboximages/'

#Merge stage 1 (blanks) and stage 2 (species)
full_ensemble = pd.merge(run_ensemble_stage_1(), run_ensemble_stage_2(),
         on = 'image_group_id')

#Stage 3: Counts and bboxes
counts_bboxes = run_ensemble_stage_3()


#Merge All Stages to produce 2 tables (or csv)

##Full ensemble with counts (by event)
full_ensemble = pd.merge(full_ensemble, counts_bboxes,
                    on = 'image_group_id', how = 'left')
full_ensemble = full_ensemble.rename(columns={'ensemble_pred': 'blank',
                             'event_final_pred': 'species'})
full_ensemble['blank'] = full_ensemble['blank'].apply(lambda x: True if x == 'blank' else False)

'''
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
    url_image_1.append(image_dir + img_grp + image_appendices[0])
    url_image_2.append(image_dir + img_grp + image_appendices[1])
    url_image_3.append(image_dir + img_grp + image_appendices[2])

    #Bbox images
    url_image_1_bbox.append(bbox_image_dir + img_grp + image_appendices[0])
    url_image_2_bbox.append(bbox_image_dir + img_grp + image_appendices[1])
    url_image_3_bbox.append(bbox_image_dir + img_grp + image_appendices[2])



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
'bboxes': final_bbox_list,
'event_final_topk_conf': final_top_3_list
})

event_images_table.insert(0, 'event_id', range(0, len(event_images_table)))

#Load date
now = datetime.now() # current date and time
date_string = now.strftime("%m/%d/%Y")
event_images_table['load_date'] = date_string

event_images_table.to_csv('../results/event_images_table_top3.csv', index = False)

#Test to Ground truth
truth_file = pd.read_csv('../data/test_labels4-2.csv')

truth_pred_df = pd.merge(event_images_table,truth_file,
         left_on = 'image_group_id',
         right_on = 'TRIGGER_ID',
         how = 'left')

#Filter out Other and Blank
truth_pred_df_no_other = truth_pred_df[truth_pred_df['CLASS_SPECIES_RESTATED'] != 'other']


#Confusion Matrix
figure(figsize=(14, 10), dpi=80)

y_test= truth_pred_df_no_other['CLASS_SPECIES_RESTATED']
label_list = np.unique(y_test)

pred= truth_pred_df_no_other['species_name']

cm = confusion_matrix(y_test, pred, labels = label_list)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues");  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel(
    'True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(label_list); ax.yaxis.set_ticklabels(label_list);
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)

plt.show()


#Classification Report
y_true = truth_pred_df_no_other['CLASS_SPECIES_RESTATED']
y_pred = truth_pred_df_no_other['species_name']



print(classification_report(y_true, y_pred, target_names=label_list))

#Writing bounding box images
