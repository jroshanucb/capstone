import cv2
import torch
from PIL import Image
import os
import pandas as pd
import numpy as np
from datetime import datetime

from pathlib import Path
import math

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

def yolo_inference(img_directory, weights_path, imgsz):

    # Model
    model = torch.hub.load('ultralytics/yolov5','custom',
path= weights_path, force_reload=True)

    #Images
    imgs = []
    img_names = []

    for img_name in os.listdir(img_directory):
        if img_name[-4:] == '.jpg' or img_name[-4:] == 'jpeg':
            img = cv2.imread(img_directory+img_name)[:, :, ::-1]
            imgs.append(img)
            img_names.append(img_name)


    batch_size = 1000
    num_images = len(img_names)
    num_batches = int(math.ceil(num_images/batch_size))

    print("Running inference on {} images in {} batches.".format(num_images, num_batches))
    # Inference in batches

    first_batch = True
    for batch_num in range(1, num_batches+1):
        img_batch = imgs[(batch_size*(batch_num - 1)) :(batch_size*batch_num)]
        img_name_batch = img_names[(batch_size*(batch_num - 1)) :(batch_size*batch_num)]

        print('Running batch {}, {} images.'.format(batch_num, len(img_batch)))

        results = model(img_batch, size=imgsz)  # includes NMS

        #Combine results from all rows of batch into single pandas df
        first_row = True
        for tensor,image_name in zip(results.xyxy, img_name_batch):
            int_results_df = pd.DataFrame(np.array(tensor.cpu()))

            int_results_df['image_name'] = image_name

            if first_row == True:
                batch_results_df = int_results_df
                first_row = False
            else:
                batch_results_df = pd.concat([batch_results_df,
                                             int_results_df])
        #Combine results from all batches into single pandas df
        if first_batch == True:
                full_results_df = batch_results_df
                first_batch = False
        else:
            full_results_df = pd.concat([full_results_df,
                                         batch_results_df])

    full_results_df =full_results_df.set_axis(['xmin','ymin', 'xmax', 'ymax', 'conf', 'class', 'image_name'],
                                             axis = 1, inplace = False)

    #Blank images do not produce any results so we need to add blank rows wiht just the image names
    blank_imgs = [img for img in img_names if img not in list(full_results_df['image_name'])]
    blank_img_df = pd.DataFrame(columns = full_results_df.columns)
    blank_img_df['image_name'] = blank_imgs
    blank_img_df = blank_img_df.fillna('')

    full_results_df = pd.concat([full_results_df, blank_img_df])
    full_results_df.to_csv('yolo_full_results.csv')
    return full_results_df

def yolo_boxes_to_df(full_results_df):
    def convert_yolo_bbox(size, box):
        '''Convert result bbox format from xmin,xmax,ymin,ymax absolute values to
        x,y,w,h relative values'''
        try:
            dw = 1./size[0]
            dh = 1./size[1]
            x = (box[0] + box[1])/2.0
            y = (box[2] + box[3])/2.0
            w = box[1] - box[0]
            h = box[3] - box[2]
            x = x*dw
            w = w*dw
            y = y*dh
            h = h*dh
            coord_string = '{},{},{},{}'.format(x,y,w,h)
        except:
            coord_string = ''


        return coord_string
    full_results_df['image_bbox'] = full_results_df.apply(lambda x: convert_yolo_bbox((329,329), [x['xmin'], x['xmax'], x['ymin'], x['ymax']]), axis = 1)

    return full_results_df

def yolo_spec_conf_bbox_formatting(full_results_df):
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
    image_id_1_bbox = []
    image_id_2_bbox = []
    image_id_3_bbox = []

    current_image = ''
    current_event = ''
    current_image_appendix = ''

    int_image_species = ''
    int_image_conf = ''
    int_image_bbox = ''

    i = 0

    for row, value in full_results_df.sort_values(by = 'image_name').iterrows():

        #Get current image and event names
        next_image = value['image_name']
        next_image_appendix = next_image.split('.')[0][-1]

        next_event = img_name_to_event(next_image)

        if next_event != current_event:
            end_of_event = True
        if next_image != current_image:
            end_of_image = True

        if end_of_image == True:
            #If first event of dataframe, i
            if i == 1 or i == 0:
                image_id_1.append(current_image_appendix)
                image_id_1_species_name.append(int_image_species)
                image_id_1_conf.append(int_image_conf)
                image_id_1_bbox.append(int_image_bbox)
            if i == 2:
                image_id_2.append(current_image_appendix)
                image_id_2_species_name.append(int_image_species)
                image_id_2_conf.append(int_image_conf)
                image_id_2_bbox.append(int_image_bbox)

            if i == 3:
                image_id_3.append(current_image_appendix)
                image_id_3_species_name.append(int_image_species)
                image_id_3_conf.append(int_image_conf)
                image_id_3_bbox.append(int_image_bbox)

            end_of_image = False
            i += 1

            int_image_species = ''
            int_image_conf = ''
            int_image_bbox = ''

        if end_of_event == True:
            if i == 2 or i ==1:
                image_id_2.append('')
                image_id_2_species_name.append('')
                image_id_2_conf.append('')
                image_id_2_bbox.append('')

                image_id_3.append('')
                image_id_3_species_name.append('')
                image_id_3_conf.append('')
                image_id_3_bbox.append('')

            elif i == 3:
                image_id_3.append('')
                image_id_3_species_name.append('')
                image_id_3_conf.append('')
                image_id_3_bbox.append('')
            end_of_event = False
            i = 1

            image_group_id.append(current_event)

        #Setting new current values
        #If image has already registed a species, need a seperator between next entry
        if len(int_image_species) == 0:
            spec_conf_pre = ''
            bbox_pre = ''
        else:
            spec_conf_pre = ','
            bbox_pre = ';'
        spec_to_add = spec_conf_pre + value['species_name']
        conf_to_add = spec_conf_pre + str(value['conf'])
        bbox_to_add = bbox_pre + value['image_bbox']

        int_image_species += spec_to_add
        int_image_conf += conf_to_add
        int_image_bbox += bbox_to_add


        current_image = next_image
        current_event = next_event
        current_image_appendix = next_image_appendix

    image_group_id.append(current_event)


    if i <= 3:
        image_id_3.append(current_image_appendix)
        image_id_3_species_name.append(int_image_species)
        image_id_3_conf.append(int_image_conf)
        image_id_3_bbox.append(int_image_bbox)

    if i <= 2:
        image_id_2.append(current_image_appendix)
        image_id_2_species_name.append(int_image_species)
        image_id_2_conf.append(int_image_conf)
        image_id_2_bbox.append(int_image_bbox)
    if i <= 1:
        image_id_1.append(current_image_appendix)
        image_id_1_species_name.append(int_image_species)
        image_id_1_conf.append(int_image_conf)
        image_id_1_bbox.append(int_image_bbox)

    formatted_yolo = pd.DataFrame({'image_group_id':image_group_id,
    'image_id_1':image_id_1,
    'image_id_2':image_id_2,
    'image_id_3':image_id_3,
    'image_id_1_species_name': image_id_1_species_name,
    'image_id_2_species_name':image_id_2_species_name,
    'image_id_3_species_name' :image_id_3_species_name,
    'image_id_1_conf': image_id_1_conf,
    'image_id_2_conf': image_id_2_conf,
    'image_id_3_conf': image_id_3_conf,
    'image_id_1_bbox': image_id_1_bbox,
    'image_id_2_bbox': image_id_2_bbox,
    'image_id_3_bbox': image_id_3_bbox})

    formatted_yolo = formatted_yolo[1:]

    return formatted_yolo

def yolo_count_blank_detect_formatting(formatted_yolo, model_id):
    def count_species(species_string):
        species_list = species_string.split(',')
        if species_list[0] == '':
            return 0
        else:
            return len(species_list)

    for image in range(1,4):
        formatted_yolo['image_id_{}_count'.format(image)] = formatted_yolo['image_id_{}_species_name'.format(image)].apply\
                                                        (lambda x:count_species(x))

    for image in range(1,4):
        formatted_yolo['image_id_{}_blank'.format(image)] = formatted_yolo['image_id_{}_species_name'.format(image)].apply\
                                                            (lambda x:True if x == '' else False)

    #Model ID
    formatted_yolo['model_id'] = model_id

    return formatted_yolo

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

d = {'img_id':[], 'blurry':[], 'blurry_index':[]}


def blur_processing(img_directory, formatted_yolo, run_blur):
    # loop over the input images
    if run_blur == True:
        for img in os.listdir(img_directory):
            if img[-4:] == '.jpg' or img[-4:] == 'jpeg':
                # load the image, convert it to grayscale, and compute the
                # focus measure of the image using the Variance of Laplacian
                # method
                image = cv2.imread(img_directory + img)
                filename = img

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                fm = variance_of_laplacian(gray)
                threshold = 100
                text = "Not Blurry"
                # if the focus measure is less than the supplied threshold,
                # then the image should be considered "blurry"
                if fm < threshold:
                    text = "Blurry"
                    d['img_id'].append(filename)
                    d['blurry'].append(True)
                    d['blurry_index'].append(fm)

                else:
                    d['img_id'].append(filename)
                    d['blurry'].append(False)
                    d['blurry_index'].append(fm)

        blur_df = pd.DataFrame(d)

        blur_df['image_group_id'] = blur_df['img_id'].apply(lambda x: img_name_to_event(x))
        blur_df['img_appendix'] = blur_df['img_id'].apply(lambda x: x.split('.')[0][-1])

        for i in range(1,4):
            formatted_yolo = pd.merge(formatted_yolo, blur_df[['image_group_id', 'img_appendix', 'blurry']],
                     how = 'left',
                     left_on = ['image_group_id', 'image_id_{}'.format(i)],
                     right_on = ['image_group_id', 'img_appendix'])

            formatted_yolo['image_id_{}_detectable'.format(i)] = formatted_yolo['blurry']
            formatted_yolo = formatted_yolo.drop(columns = ['img_appendix', 'blurry'])

    else:
        for i in range(1,4):
            formatted_yolo['image_id_{}_detectable'.format(i)] = False

    return formatted_yolo


def run_format_yolo(img_directory, weight, imgsz, labels, model_id, run_blur = True):

    print('Running Yolo, Model ID {}'.format(model_id))
    full_results_df = yolo_inference(img_directory, weight, imgsz)
    full_results_df = yolo_boxes_to_df(full_results_df)
    full_results_df = codes_to_labels(full_results_df, labels)

    formatted_yolo = yolo_spec_conf_bbox_formatting(full_results_df)
    formatted_yolo = yolo_count_blank_detect_formatting(formatted_yolo, model_id)

    formatted_yolo = blur_processing(img_directory, formatted_yolo, run_blur)

    return formatted_yolo
