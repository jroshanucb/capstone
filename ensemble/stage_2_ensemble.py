import os
import random
import shutil, os
import pandas as pd
import numpy as np
from collections import Counter

from os import listdir
from os.path import isfile, join


#Read lines from txt results file
top_path = '../results/'
output_file = 'full_model_output.csv'

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

"""
output:
event level: event_id, species class name, count, bounding boxes, top3 dict (species name: confidence)
scale yolo top1 by 1.5x

output format:

  { event_id: ######
      {
        species_class: 0
        species_name: deer
        count: 1
        bbox: [
          [x, y, w, h]
        ]
        conf: 0.79
        top3: {
          deer: 0.79,
          elk: 0.4
          fox: 0.1
        }
      }

  }
"""

def create_species_conf_dict(x, y, model):
  """
  zip class predictions and confidence scores together into a dictionary
  at the image level

  model 3 does not have sorted confidence scores
  model 3 can also have classes appear more than once per image
  """
  if model == 3:
    new_dictionary = {}

    if isinstance(x, float):
      species_list = ['blank']
      conf_list = [str(0.99)]
    else:
      species_list = list(x.split(","))
      conf_list = list(y.split(","))

    new_dictionary = {}
    for key, value in zip(species_list, conf_list):
      if key in new_dictionary:
        new_dictionary[key].append(value)
      else:
        new_dictionary[key] = [value]

    ## sort dictionary in descending order according to value
    intermed_dict = {}
    for key, value in new_dictionary.items():
      intermed_dict[key] = sorted(value, reverse=True)

    ## trigger turns on if there are classes with multiple appearances

    trigger = ''
    if len(new_dictionary) >= 2:
      for key, value in new_dictionary.items():
        if len(value) > 1:
          trigger = 1
        else:
          continue

    if trigger:
      ## if there are multiple classes with multiple appearances and are tied in length
      ## we need a tie-breaker...default to class with higher max confidence
      ## python's max function defaults to first encounter for tie-breaker...not good enough for us

      ## collect the length of all lists in dictionary
      lists_length = []
      for key, value in new_dictionary.items():
        lists_length.append(len(value))

      lists_length = sorted(lists_length, reverse=True)

      ## since the list is already sorted, we can just check if the first two lists are equal
      ## if equal, then there is a tie
      tie = ''
      if lists_length[0] == lists_length[1]:
        tie = 1

      ## if there is a tie, then take the max conf score
      if tie:
        int_dict = {}
        for key, value in new_dictionary.items():
          if len(value) > 1:
            int_dict[key] = max(value)

        best_class = sorted(int_dict, key=int_dict.get, reverse=True)[:1][0]
        max_conf = int_dict[best_class]
        output_dict = {best_class: max_conf}

      else:
        ## default to class with most appearances
        best_class = max(intermed_dict, key= lambda x: len(set(intermed_dict[x])))
        max_conf = max(intermed_dict[best_class])
        output_dict = {best_class: max_conf}
    else:
      ## default to class with highest confidence score
      best_class = sorted(intermed_dict, key=intermed_dict.get, reverse=True)[:1]
      max_conf = max(intermed_dict[best_class[0]])
      output_dict = {best_class[0]: max_conf}

    return output_dict

  else:
  ## works with sorted confidence scores, modelid = 4
    if isinstance(x, float):
      species_list = ['blank']
      conf_list = [str(0.99)]
      return dict(zip(species_list, conf_list))
    else:
      species_list = list(x.split(","))
      conf_list = list(y.split(","))
      return dict(zip(species_list, conf_list))


def merge_species_conf_dict_top3(x, y, z):
  """
  merge image-level dictionaries into event level
  only gets the top prediction for each image
  """
  if x is None:
    x = {'blank': str(99)}
  if y is None:
    y = {'blank': str(99)}
  if z is None:
    z = {'blank': str(99)}

  dict_list = [list(x.items())[0],list(y.items())[0],list(z.items())[0]]
  #print(dict_list)
  preds_dict = {}

  for item in dict_list:
    if item is None:
      pass
    else:
      if item[0] in preds_dict:
        preds_dict[item[0]].append(float(item[1]))
      else:
        preds_dict[item[0]] = [float(item[1])]


  return preds_dict

def get_topk(x, k):
  """
  get top k entries in dictionary according to confidence score
  if blanks are in the dictionary, modify to remove blank if more than 1 class available
  """
  if 'blank' in x and len(x) >= 2:
    temp_dict = x.copy()
    del temp_dict['blank']
    topk_event_dict = {}
    top_ind = sorted(temp_dict, key=temp_dict.get, reverse=True)[:k]

    # conf_scores = {}
    # for i in top_ind:
    #   conf_scores[i] = max(x[i])
    # conf_scores['blank'] = str(0.99)

  else:
    topk_event_dict = {}
    top_ind = sorted(x, key=x.get, reverse=True)[:k]

  conf_scores = {}
  for i in top_ind:
    conf_scores[i] = max(x[i])

  return conf_scores

def get_pred_from_top3(consol_dict):

  ## if all 3 predictions are different classes, defer to class with highest confidence
  ## result cannot be blank though
  if len(consol_dict) == 3 and 'blank' not in consol_dict:
    return get_topk(consol_dict, 1)

  elif len(consol_dict) == 3 and 'blank' in consol_dict:
    temp_dict = consol_dict.copy()
    del temp_dict['blank']
    return get_topk(temp_dict, 1)

  ## exception to the below rule is if there are blanks
  elif len(consol_dict) == 2 and 'blank' in consol_dict:
    temp_dict = consol_dict.copy()
    del temp_dict['blank']
    max_key = max(temp_dict, key= lambda x: len(set(temp_dict[x])))
    return {max_key: max(temp_dict[max_key])}

  ## if there is an even number of predictions (2), defer to class with more appearances (has a longer list)
  elif len(consol_dict) == 2:
    max_key = max(consol_dict, key= lambda x: len(set(consol_dict[x])))
    return {max_key: max(consol_dict[max_key])}

  ## if there is only one class of predictions, return the class with highest confidence
  else:
    return get_topk(consol_dict, 1)

def merge_species_conf_dict(x, y, z):
  """
  combine full confidence dictionaries across images in an event
  """
  dict_list = [x,y,z]
  #print(dict_list)
  preds_dict = {}

  # for item in dict_list:
  #   if item is None:
  #     pass
  #   else:
  #     for key, value in item.items():
  #       if key in preds_dict:
  #         preds_dict[key].append(value)
  #       else:
  #         preds_dict[key] = [value]
  for item in dict_list:
    #print(item)
    for key, value in item.items():
      if key in preds_dict:
        preds_dict[key].append(value)
      else:
        preds_dict[key] = [value]

  return preds_dict

"""
before we combine scores, we should weight the predictions between model_id3 and model_id4
to slightly bias the scores towards model_id3

80/67 = 1.2

multiply topk_conf_3 by 1.2
divide topk_conf_4 by 1.2
"""

def scale_model_id3(x):
  ## x is a dictionary
  intermed_dict = {key: float(value) * 1.2 for key, value in x.items()}
  ## cap maximum possible score at 0.99
  output_dict = {key: (0.99 if float(value) > 1 else float(value)) for key, value in intermed_dict.items()}
  return output_dict

def scale_model_id4(x):
  ## x is a dictionary
  return {key: float(value) / 1.2 for key, value in x.items()}

"""
output final topk dictionary of species predictions and their confidence scores
"""

def combine_topk_conf(x, y):
  dict_list = [x,y]
  #print(dict_list)
  preds_dict = {}

  for item in dict_list:
    if item is None:
      pass
    else:
      for key, value in item.items():
        if key in preds_dict:
          preds_dict[key].append(value)
        else:
          preds_dict[key] = [value]

  return preds_dict

def get_final_topk(x, k):
  topk_event_dict = {}
  top_ind = sorted(x, key=x.get, reverse=True)[:k]
  conf_scores = {}
  for i in top_ind:
    conf_scores[i] = max(x[i])

  return conf_scores

"""
output final event prediction, conf score
implement blank rule
if highest prediction score is less than 30%, defer to blank
"""
def output_final_pred_species(x):

  ## if blank is in the top-3 predictions, it's probably a top prediction and could overrule
  ## the animal predcitions, resulting in too many false negatives
  if (len(x) == 3 and 'blank' in x) or (len(x) == 2 and 'blank' in x):

    ## temporary dictionary with blank class removed
    temp_dict = x.copy()
    del temp_dict['blank']

    ## get best available class that's not blank based on temporary dictionary
    max_class = max(temp_dict, key=x.get)
    max_conf = temp_dict[max_class]

    ## set a threshold of 0.5
    if max_conf <= 0.5:
      ## defer to blank...use original dictionary
      intermed_dict = {key: float(value) for key, value in x.items()}
      top_ind = sorted(intermed_dict, key=intermed_dict.get, reverse=True)[:1]
      return top_ind[0]

    else:
      ## use temporary dictionary
      intermed_dict = {key: float(value) for key, value in temp_dict.items()}
      top_ind = sorted(intermed_dict, key=intermed_dict.get, reverse=True)[:1]
      return top_ind[0]

  else:
    intermed_dict = {key: float(value) for key, value in x.items()}
    top_ind = sorted(intermed_dict, key=intermed_dict.get, reverse=True)[:1]
    #print(top_ind)
    return top_ind[0]

def output_final_pred_conf(x, dict_col):
  # if not x:
  #   return None
  # else:
  #   intermed_dict = {key: float(value) for key, value in x.items()}
  #   top_ind = sorted(intermed_dict, key=intermed_dict.get, reverse=True)[:1]
  #   #print(top_ind)
  #   return x[top_ind[0]]
  return dict_col[x]

def yolo_final_pred(top_pred):
    if top_pred == 'blank':
        return 'blank'
    else:
        return list(top_pred.keys())[0]

def run_ensemble_stage_2(modelsz):

    sample_input = pd.read_csv(top_path + output_file)

    df_model_id3 = sample_input[sample_input.model_id == 3]


    df_model_id3['img1_species_conf_dict'] = df_model_id3.apply(lambda x: create_species_conf_dict(x.image_id_1_species_name, x.image_id_1_conf, x.model_id), axis=1)
    df_model_id3['img2_species_conf_dict'] = df_model_id3.apply(lambda x: create_species_conf_dict(x.image_id_2_species_name, x.image_id_2_conf, x.model_id), axis=1)
    df_model_id3['img3_species_conf_dict'] = df_model_id3.apply(lambda x: create_species_conf_dict(x.image_id_3_species_name, x.image_id_3_conf, x.model_id), axis=1)
    df_model_id3['consol_dict'] = df_model_id3.apply(lambda x: merge_species_conf_dict_top3(x.img1_species_conf_dict, x.img2_species_conf_dict, x.img3_species_conf_dict), axis=1)
    df_model_id3['top_pred'] = df_model_id3.apply(lambda x: get_pred_from_top3(x.consol_dict), axis=1)
    df_model_id3['top3_dict'] = df_model_id3.apply(lambda x: get_topk(merge_species_conf_dict(x.img1_species_conf_dict, x.img2_species_conf_dict, x.img3_species_conf_dict),3), axis=1)
    if modelsz == 'small':
        df_model_id3 =  df_model_id3.rename(columns = {'consol_dict': 'consol_dict_model_3'})
        df_model_id3['consol_dict_model_4'] = ''
        df_model_id3['event_final_topk_conf'] = ''
        df_model_id3['event_final_pred'] = df_model_id3['top_pred'].apply(lambda top_pred: yolo_final_pred(top_pred))

        return df_model_id3[['image_group_id','consol_dict_model_3', 'consol_dict_model_4', 'event_final_topk_conf', 'event_final_pred']]

    if modelsz in ['medium', 'large']:
        df_model_id4 = sample_input[sample_input.model_id == 4]
        df_model_id4['img1_species_conf_dict'] = df_model_id4.apply(lambda x: create_species_conf_dict(x.image_id_1_species_name, x.image_id_1_conf, x.model_id), axis=1)
        df_model_id4['img2_species_conf_dict'] = df_model_id4.apply(lambda x: create_species_conf_dict(x.image_id_2_species_name, x.image_id_2_conf, x.model_id), axis=1)
        df_model_id4['img3_species_conf_dict'] = df_model_id4.apply(lambda x: create_species_conf_dict(x.image_id_3_species_name, x.image_id_3_conf, x.model_id), axis=1)
        df_model_id4['consol_dict'] = df_model_id4.apply(lambda x: merge_species_conf_dict_top3(x.img1_species_conf_dict, x.img2_species_conf_dict, x.img3_species_conf_dict), axis=1)
        df_model_id4['top_pred'] = df_model_id4.apply(lambda x: get_pred_from_top3(x.consol_dict), axis=1)
        df_model_id4['top3_dict'] = df_model_id4.apply(lambda x: get_topk(merge_species_conf_dict(x.img1_species_conf_dict, x.img2_species_conf_dict, x.img3_species_conf_dict),3), axis=1)


        """
        merge model_id3 and model_id4 at the event level
        the final prediction dictionaries from both models can be on the same row
        """
        df_merge = pd.merge(df_model_id3, df_model_id4, how='inner', on="image_group_id", suffixes=('_model_3', '_model_4'))
        df_merge = df_merge[['image_group_id', 'consol_dict_model_3', 'top_pred_model_3', 'top3_dict_model_3', 'consol_dict_model_4', 'top_pred_model_4', 'top3_dict_model_4']]
        #df_merge.head()

        df_merge['topk_conf_3_scaled'] = df_merge.apply(lambda x: scale_model_id3(x.top3_dict_model_3), axis=1)
        df_merge['topk_conf_4_scaled'] = df_merge.apply(lambda x: scale_model_id4(x.top3_dict_model_4), axis=1)

        df_merge['event_final_topk_conf'] = df_merge.apply(lambda x: get_final_topk(combine_topk_conf(x.topk_conf_3_scaled, x.topk_conf_4_scaled),3), axis=1)
        df_merge['event_final_pred'] = df_merge.apply(lambda x: output_final_pred_species(x.event_final_topk_conf), axis=1)
        df_merge['event_final_pred_conf'] = df_merge.apply(lambda x: output_final_pred_conf(x.event_final_pred, x.event_final_topk_conf), axis=1)

        #df_merge.to_csv('../results/merged_stage_2_pred_conf.csv', index = False)
        return df_merge[['image_group_id','consol_dict_model_3', 'consol_dict_model_4', 'event_final_topk_conf', 'event_final_pred']]
