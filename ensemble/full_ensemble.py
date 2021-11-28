import pandas as pd
import numpy as np

from stage_1_ensemble import run_ensemble_stage_1
from stage_2_ensemble import run_ensemble_stage_2
from stage_3_ensemble import run_ensemble_stage_3

#Merge stage 1 (blanks) and stage 2 (species)
full_ensemble = pd.merge(run_ensemble_stage_1(), run_ensemble_stage_2(),
         on = 'image_group_id')

#Stage 3: Counts and bboxes
counts_bboxes = run_ensemble_stage_3()
# counts = counts_bboxes.groupby('image_group_id').max()[['final_count']].reset_index()
# bboxes = counts_bboxes[['image_group_id', 'image_id', 'final_bbox']]

#Merge All Stages to produce 2 tables (or csv)

##Full ensemble with counts (by event)
full_ensemble = pd.merge(full_ensemble, counts_bboxes,
                    on = 'image_group_id', how = 'left')
full_ensemble = full_ensemble.rename(columns={'ensemble_pred': 'blank',
                             'event_final_pred': 'species'})
full_ensemble['blank'] = full_ensemble['blank'].apply(lambda x: True if x == 'blank' else False)
full_ensemble.to_csv('../results/full_ensemble.csv', index = False)

##BBoxes by image
# bboxes.to_csv('../results/bbox_by_image.csv', index = False)
