import pandas as pd
import numpy as np

from stage_1_ensemble import run_ensemble_stage_1

from stage_2_ensemble import run_ensemble_stage_2

full_ensemble = pd.merge(run_ensemble_stage_1(), run_ensemble_stage_2(),
         on = 'image_group_id')

full_ensemble = full_ensemble.rename(columns={'ensemble_pred': 'blank',
                             'event_final_pred': 'species'})

full_ensemble['blank'] = full_ensemble['blank'].apply(lambda x: True if x == 'blank' else False)
