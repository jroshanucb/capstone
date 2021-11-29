!pip install timm
!pip install efficientnet_pytorch

import os
import pandas as pd
from tqdm.notebook import tqdm
tqdm().pandas()
import shutil


from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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


# Number of classes in the dataset
num_classes = 11

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True



from efficientnet_pytorch import EfficientNet
import timm
## https://discuss.pytorch.org/t/resnet-last-layer-modification/33530
## https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088

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

      ### train/val vs test are from different distributions
      ### https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/39
      ### https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/16
      # for m in model_ft.modules():
      #     if isinstance(m, nn.BatchNorm2d):
      #       m.track_running_stats=False

      input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

### transform image input to tensor
input_size = 224
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

"""
 If feature_extract = False, the model is finetuned and all model parameters are updated.
 If feature_extract = True, only the last layer parameters are updated, the others remain fixed.

 In finetuning, we start with a pretrained model and update all of the model’s parameters for our new task, in essence retraining the whole model.
    feature_extract = False
    use_pretrained = True
 In feature extraction, we start with a pretrained model and only update the final layer weights from which we derive predictions.
    feature_extract = True
    use_pretrained = True
 Train from scratch:
    feature_extract = False
    use_pretrained = False
"""


class_names = ['bear',
 'blank',
 'cottontail_snowshoehare',
 'coyote',
 'deer',
 'elk',
 'foxgray_foxred',
 'opossum',
 'raccoon',
 'turkey',
 'wolf']


## load model weights
checkpoint = torch.load(Path('/content/gdrive/My Drive/Colab Notebooks/w210_capstone/yolo_splits_torch_model_runs/efficientnetb5_100epochs_finetuned_model_yolosplits4_BasePlusBlank'))
model_name = 'efficientnetb5'
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False) #change True/False
model_ft.load_state_dict(checkpoint)
model_ft.eval()

# image_path = ''

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

### batch inference for directory of images

def perform_inference_batch(img_dir):
    classifications = []

    with torch.no_grad():
        with tqdm(total = len(img_dir)) as pbar:
            for image in img_dir:
                pbar.set_description("processing {}".format(image))
                image_inst = Image.open(Path(image)).convert('RGB')
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

                classifications.append(classification)
                pbar.update(1)

    return classifications
