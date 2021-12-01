import cv2
import torch
from PIL import Image
import os
import pandas as pd
import numpy as np
from datetime import datetime

img_directory = '/Users/sleung2/Documents/MIDS Program/Capstone_local/snapshot_wisconsin/all/yolo_splits4.1/test/images/'

##Labels

#Stage 1
stage_1_labels = pd.DataFrame(['animal']).sort_values(0)
stage_1_labels = stage_1_labels.rename(columns = {0: 'species'})
stage_1_labels.insert(0, 'label', range(0, len(stage_1_labels)))

#Stage 2
stage_2_labels = pd.DataFrame(['foxgray_foxred',
              'cottontail_snowshoehare',
              'raccoon',
              'opossum',
              'turkey',
              'bear',
              'elk',
              'deer',
              'coyote',
              'wolf']).sort_values(0)
stage_2_labels = stage_2_labels.rename(columns = {0: 'species'})
stage_2_labels.insert(0, 'label', range(0, len(stage_2_labels)))

def yolo_inference(img_directory, weights_path):

    # Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

    #Images
    imgs = []
    img_names = []

    i = 1
    for img_name in os.listdir(img_directory):
        if i == 100:
            break
        img = cv2.imread(img_directory+img_name)[:, :, ::-1]
        imgs.append(img)
        img_names.append(img_name)
        i+=1

    print("Running inference on {} images".format(len(img_names)))
    # Inference
    results = model(imgs, size=329)  # includes NMS

    #Combine results from all images into single pandas df
    first = True
    for tensor,image_name in zip(results.xyxy, img_names):
        int_results_df = pd.DataFrame(np.array(tensor))

        int_results_df['image_name'] = image_name

        if first == True:
            full_results_df = int_results_df
            first = False
        else:
            full_results_df = pd.concat([full_results_df,
                                         int_results_df])

    full_results_df =full_results_df.set_axis(['xmin','ymin', 'xmax', 'ymax', 'conf', 'class', 'image_name'],
                                         axis = 1, inplace = False)

    #Blank images do not produce any results so we need to add blank rows wiht just the image names
    blank_imgs = [img for img in img_names if img not in list(full_results_df['image_name'])]
    blank_img_df = pd.DataFrame(columns = full_results_df.columns)
    blank_img_df['image_name'] = blank_imgs
    blank_img_df = blank_img_df.fillna('')

    full_results_df = pd.concat([full_results_df, blank_img_df])

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
    if i <= 3:
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
    #Load date
    now = datetime.now() # current date and time
    date_string = now.strftime("%m/%d/%Y")
    formatted_yolo['load_date '] = date_string

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


#EfficientNet
def get_speciesname_from_id(id, phase):

    if phase == 'phase1':
        speciesList = ['animal', 'blank']
    else:
        speciesList =  ['bear', 'blank', 'cottontail_snowshoehare', 'coyote', 'deer', 'elk', 'foxgray_foxred', 'opossum', 'raccoon', 'turkey', 'wolf']
    idx = int(id)
    if idx > 10 or idx < 0:
        speciesName = 'other'
    else:
        speciesName = speciesList[idx]
    return speciesName

def load_effnet_json(output_json, phase = 'phase2'):
    effnet_json = {}

    with open(output_json) as json_file:
        data = json.load(json_file)

    data = data['{}_classification_results'.format(phase)]
    for dict_list in data:
        value = dict_list
        newKey = value['id']
        effnet_json[newKey] = {}
        class_list = ""
        conf_list = ""

        for key2, value2 in value['conf_dict'].items():
            class_list += get_speciesname_from_id(key2, phase) + ","
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
    image_id_1_bbox = []
    image_id_2_bbox = []
    image_id_3_bbox = []

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

    ['image_id_1_blank']

    for image in range(1,4):
        formatted_effnet['image_id_{}_count'.format(image)] = 0

    for image in range(1,4):
        formatted_effnet['image_id_{}_bbox'.format(image)] = 0



    for image in range(1,4):
        if model_id == 2:
            formatted_effnet['image_id_{}_blank'.format(image)] = formatted_effnet['image_id_{}_species_name'.format(image)].apply\
                                                            (lambda x:True if x == 'blank' else False)
        else:
            formatted_effnet['image_id_{}_blank'.format(image)] = formatted_effnet['image_id_{}_species_name'.format(image)].apply\
                                                            (lambda x:True if x == '' else False)

    for image in range(1,4):
        formatted_effnet['image_id_{}_detectable'.format(image)] = False

    formatted_effnet['model_id'] = model_id

    return formatted_effnet

stage2_output_json = '../results/JSON_txt_outputs/phase2_efficientnetb5_yolo_splits4-1_classifications_basePlusblanks.json'

stage2_effnet_dict = load_effnet_json(stage2_output_json, 'phase2')
stage2_formatted_effnet = format_effnet(stage2_effnet_dict, 4)


stage1_output_json = '../results/JSON_txt_outputs/phase1_efficientnetb0_classifications_yolosplits_4-1.json'

stage1_effnet_dict = load_effnet_json(stage1_output_json, 'phase1')
stage1_formatted_effnet = format_effnet(stage1_effnet_dict, 2)
