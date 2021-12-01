import os
import pandas as pd
import numpy as np
import json

#Helper functions
def codes_to_labels(full_results_df, labels):

    label_dict = labels.set_index('label').to_dict()['species']
    full_results_df['species_name'] = full_results_df['class'].map(label_dict)
    full_results_df = full_results_df.fillna('')

    return full_results_df

def img_name_to_event(img):
    if '_' in img:
        event_name = img.split('_')[0]
    else:
        event_name = img.split('.')[0][:-1]

    return event_name

def get_speciesname_from_id(id, labels):

    speciesList = list(labels['species'])
    idx = int(id)
    if idx > 10 or idx < 0:
        speciesName = 'other'
    else:
        speciesName = speciesList[idx]
    return speciesName

def load_effnet_json(output_json, model_id, labels):
    effnet_json = {}

    with open(output_json) as json_file:
        data = json.load(json_file)

    if model_id == 2:
        phase = 'phase1'
    else:
        phase = 'phase2'
    data = data['{}_classification_results'.format(phase)]
    for dict_list in data:
        value = dict_list
        newKey = value['id']
        effnet_json[newKey] = {}
        class_list = ""
        conf_list = ""

        for key2, value2 in value['conf_dict'].items():
            class_list += get_speciesname_from_id(key2, labels) + ","
            conf_list += str(value2[0]) + ","

        effnet_json[newKey]['Class'] = class_list[:-1]
        effnet_json[newKey]['Conf'] = conf_list[:-1]

        # print(effnet_json)
    return effnet_json


def format_effnet(effnet_dict, model_id):
    '''Convert effnet dict to pandas df'''
    event_list = []
    image_list = []
    class_list = []
    conf_list = []

    for key, value in effnet_dict.items():
        event_list.append(img_name_to_event(key))
        image_list.append(key)
        class_list.append(value['Class'])
        conf_list.append(value['Conf'])

    effnet_int_df = pd.DataFrame({'event':event_list,
                 'image':image_list,
                 'class': class_list,
                 'conf': conf_list}).sort_values(by='image')

    image_group_id = []
    image_id_1 = []
    image_id_2 = []
    image_id_3 = []
    image_id_1_species_name = []
    image_id_2_species_name = []
    image_id_3_species_name = []
    image_id_1_conf = []
    image_id_2_conf = []
    image_id_3_conf = []

    for event in effnet_int_df['event'].unique():
        image_group_id.append(event)
        event_effnet = effnet_int_df[effnet_int_df['event'] == event]

        i = 1
        for row, values in event_effnet.iterrows():
            image_appendix = values['image'].split('.')[0][-1]
            if model_id == 2:
                class_for_row = values['class'].split(',')[0]
                conf_for_row = values['conf'].split(',')[0]
            else:
                class_for_row = values['class']
                conf_for_row = values['conf']

            if i ==1:
                image_id_1.append(image_appendix)
                image_id_1_species_name.append(class_for_row)
                image_id_1_conf.append(conf_for_row)
            elif i == 2:
                image_id_2.append(image_appendix)
                image_id_2_species_name.append(class_for_row)
                image_id_2_conf.append(conf_for_row)
            elif i == 3:
                image_id_3.append(image_appendix)
                image_id_3_species_name.append(class_for_row)
                image_id_3_conf.append(conf_for_row)
            i+=1

        if i == 3:
            image_id_3.append('')
            image_id_3_species_name.append('')
            image_id_3_conf.append('')
        if i == 2:
            image_id_2.append('')
            image_id_2_species_name.append('')
            image_id_2_conf.append('')

            image_id_3.append('')
            image_id_3_species_name.append('')
            image_id_3_conf.append('')

    formatted_effnet = pd.DataFrame({'image_group_id':image_group_id,
        'image_id_1':image_id_1,
        'image_id_2':image_id_2,
        'image_id_3':image_id_3,
        'image_id_1_species_name': image_id_1_species_name,
        'image_id_2_species_name':image_id_2_species_name,
        'image_id_3_species_name' :image_id_3_species_name,
        'image_id_1_conf': image_id_1_conf,
        'image_id_2_conf': image_id_2_conf,
        'image_id_3_conf': image_id_3_conf})

    for image in range(1,4):
        formatted_effnet['image_id_{}_count'.format(image)] = 0

    for image in range(1,4):
        formatted_effnet['image_id_{}_bbox'.format(image)] = ''

    for image in range(1,4):
        if model_id == 2:
            formatted_effnet['image_id_{}_blank'.format(image)] = formatted_effnet['image_id_{}_species_name'.format(image)].apply\
                                                            (lambda x:True if x == 'blank' else False)
        else:
            formatted_effnet['image_id_{}_blank'.format(image)] = formatted_effnet['image_id_{}_species_name'.format(image)].apply\
                                                            (lambda x:True if x.split(',')[0] == 'blank' else False)

    for image in range(1,4):
        formatted_effnet['image_id_{}_detectable'.format(image)] = False

    formatted_effnet['model_id'] = model_id

    return formatted_effnet

def run_format_effnet(json_path, labels, model_id):

    print('Running EfficientNet, Model ID {}'.format(model_id))
    stage2_effnet_dict = load_effnet_json(json_path, model_id, labels)
    stage2_formatted_effnet = format_effnet(stage2_effnet_dict, model_id)

    return stage2_formatted_effnet
