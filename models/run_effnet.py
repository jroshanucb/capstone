#Dependencies
#! pip install timm
#! pip install efficientnet_pytorch

from __future__ import print_function
from __future__ import division

import os
import pandas as pd
import numpy as np
import json

#from tqdm.notebook import tqdm
#tqdm().pandas()
import shutil



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

from PIL import Image
from pathlib import Path
from efficientnet_pytorch import EfficientNet
import timm

'''
MODEL Running
'''
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "efficientnetb5":
      if use_pretrained == True:
        model_ft = EfficientNet.from_pretrained('efficientnet-b5')
      else:
        model_ft = EfficientNet.from_name('efficientnet-b5')

      set_parameter_requires_grad(model_ft, feature_extract)
      num_ftrs = model_ft._fc.in_features
      #model_ft._fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

      model_ft._fc = nn.Linear(num_ftrs, num_classes)
      input_size = 224

    elif model_name == "efficientnetb0":
      if use_pretrained == True:
        model_ft = EfficientNet.from_pretrained('efficientnet-b0')
      else:
        model_ft = EfficientNet.from_name('efficientnet-b0')

      set_parameter_requires_grad(model_ft, feature_extract)
      num_ftrs = model_ft._fc.in_features
      #model_ft._fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

      model_ft._fc = nn.Linear(num_ftrs, num_classes)
      input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def perform_inference_single_image(img_path):

    with torch.no_grad():

        image_inst = Image.open(Path(img_path)).convert('RGB')
        input = data_transforms['val'](image_inst).to(device)
        input.unsqueeze_(0)

        model_ft.to(device)
        output = model_ft(input)

        ### use calibrated logits via temperature scaling
        temperature = 1.392
        output = torch.div(output, temperature)

        ## top5 pred
        sm = nn.Softmax(dim=1)
        probabilities = sm(output)

        top_5_conf, i = output.topk(5)
        prob, idx = probabilities.topk(5)

        dict_preds = {}
        itr = 0
        for x in i.cpu().numpy()[0]:
          if x in dict_preds:
            dict_preds[int(x)].append(float(prob.cpu().detach().numpy()[0][itr]))
          else:
            dict_preds[int(x)] = [float(prob.cpu().detach().numpy()[0][itr])]
          itr += 1

        best_class = max(dict_preds, key=dict_preds.get)
        species_name = class_names[best_class]
        confidence_score = dict_preds[best_class]

        classification = {
              "id": image,
              "class": int(best_class),
              "class_name": species_name,
              "conf": float(confidence_score[0]),
              "conf_dict": dict_preds
          }

    return classification

## batch inference for directory of images
def perform_inference_batch(device, img_dir, phase, weights_path, data_transforms):

    if phase == 1:
        k = 2
        temperature = 1
        num_classes = 2
        class_names = ['animal', 'blank']
        model_name = "efficientnetb0"

    elif phase == 2:
        k = 5
        temperature = 1.392
        num_classes = 11
        class_names = ['bear', 'blank', 'cottontail_snowshoehare', 'coyote', 'deer', 'elk', 'foxgray_foxred', 'opossum', 'raccoon', 'turkey', 'wolf']
        model_name = "efficientnetb5"
    else:
        print("Invalid phase number. Use either 1 or 2. Exiting...")
        return


    if str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) == 'cpu':
        checkpoint = torch.load(Path(weights_path), map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(Path(weights_path))
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False) #change True/False
    model_ft.load_state_dict(checkpoint)
    model_ft.eval()

    classifications = []

    with torch.no_grad():
        #with tqdm(total = len(img_dir)) as pbar:
            image_iteration = 1
            for image in os.listdir(img_dir):
                if image[-4:] == '.jpg' or image[-4:] == 'jpeg':
                    if image_iteration == 1:
                        print("Initiating Inference")
                    elif image_iteration % 100 == 0:
                        print("{} images done.".format(image_iteration))
#                    pbar.set_description("processing {}".format(image))
                    image_inst = Image.open(img_dir + image).convert('RGB')
                    input = data_transforms['val'](image_inst).to(device)
                    input.unsqueeze_(0)

                    model_ft.to(device)
                    output = model_ft(input)

                    ### use calibrated logits via temperature scaling
                    output = torch.div(output, temperature)

                    ## top5 pred
                    sm = nn.Softmax(dim=1)
                    probabilities = sm(output)

                    top_5_conf, i = output.topk(k)
                    prob, idx = probabilities.topk(k)

                    dict_preds = {}
                    itr = 0
                    for x in i.cpu().numpy()[0]:
                      if x in dict_preds:
                        dict_preds[int(x)].append(float(prob.cpu().detach().numpy()[0][itr]))
                      else:
                        dict_preds[int(x)] = [float(prob.cpu().detach().numpy()[0][itr])]
                      itr += 1

                    best_class = max(dict_preds, key=dict_preds.get)
                    species_name = class_names[best_class]
                    confidence_score = dict_preds[best_class]

                    classification = {
                          "id": image,
                          "class": int(best_class),
                          "class_name": species_name,
                          "conf": float(confidence_score[0]),
                          "conf_dict": dict_preds
                      }

                    classifications.append(classification)
#                    pbar.update(1)

                    image_iteration+= 1

    output_json = {'phase{}_classification_results'.format(phase): classifications}

    return output_json

def run_effnet_inference(img_directory, phase, weights_path, input_size):
    ### transform image input to tensor
    input_size = input_size
    data_transforms = {
        'train': transforms.Compose([
            #transforms.Resize((299,299)),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Resize((299,299)),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #transforms.Resize((299,299)),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_json = perform_inference_batch(device, img_directory, phase, weights_path, data_transforms)

    return output_json

'''
Formatting
'''
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

    # with open(output_json) as json_file:
    #     data = json.load(json_file)
    data = output_json

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

def run_format_effnet(img_directory, weights_path, labels, model_id):
    print('Running EfficientNet, Model ID {}'.format(model_id))

    if model_id == 2:
        phase = 1
    elif model_id == 4:
        phase = 2

    output_json = run_effnet_inference(img_directory, phase, weights_path, input_size = 329)
    stage2_effnet_dict = load_effnet_json(output_json, model_id, labels)
    stage2_formatted_effnet = format_effnet(stage2_effnet_dict, model_id)

    return stage2_formatted_effnet
