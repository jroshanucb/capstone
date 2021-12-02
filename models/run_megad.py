import os
import pandas as pd
import numpy as np
import json

#Pre-reqs
os.environ['PYTHONPATH'] += ":/content/ai4eutils"
os.environ['PYTHONPATH'] += ":/content/CameraTraps"

os.system('echo "PYTHONPATH: $PYTHONPATH"')

def run_md_inference(images_dir, output_file_path):


    os.system('!python run_tf_detector_batch.py megadetector_v4_1_0.pb "$images_dir" "$output_file_path" --recursive')

def reformat_md_json(images_dir, output_file_path):
    data = json.load(open(output_file_path))
    df = pd.DataFrame(data["images"])

    df['count'] = df['detections'].apply(lambda x: len([item for item in x]))

    """
    add event and image IDs per row
    """

    def extract_MD_imageID(x):
      return x.split(images_dir)[1]

    def extract_MD_eventID(x):

        image_id = x.split(images_dir)[1]

    if '.jpeg' in image_id:
        #image_id = x.split('.jpeg')[0]
        event_id = image_id.split('_')[0]
        return event_id
    else:
        image_id = image_id.split('.jpg')[0]
        event_id = image_id[:-1]
        return event_id

    df['event_id'] = df.apply(lambda x: extract_MD_eventID(x.file), axis=1)
    df['image_id'] = df.apply(lambda x: extract_MD_imageID(x.file), axis=1)

    def convert_coco_to_yolo(single_line ):

        yolo_list = []
        x = single_line[0]
        y = single_line[1]
        w = single_line[2]
        h = single_line[3]
        new_x = round(x + (w/2),4)
        new_y = round(y + (h/2),4)
        #new_data = [yolo_class + ' {0:.4f} {0:.4f} {0:.4f} {0:.4f}'.format(new_x,new_y,w,h)]
        #new_data = ['{}, {}, {}, {}'.format(new_x,new_y,w,h)]
        new_data = [new_x,new_y,w,h]
        yolo_list.append(new_data)
        return yolo_list[0]

    output_dict = {}

    for index, row in df.iterrows():
        img_id = row['image_id']
        event_id = row['event_id']
        detections = row['detections']

        """
        [{'bbox': [0.3783, 0.7698, 0.288, 0.2232], 'category': '1', 'conf': 0.999},
        {'bbox': [0.002641, 0.7788, 0.1323, 0.2188], 'category': '1', 'conf': 0.843}]
        """
        yolo_bbox_dict = []
        for item in detections:
        #print(item['bbox'])
        yolo_bbox = {
            "bbox": convert_coco_to_yolo(item['bbox']),
            "conf": item['conf']
        }
        yolo_bbox_dict.append(yolo_bbox)


        result_dict = {
          "img_id": img_id,
          "event_id": event_id,
          "detections": yolo_bbox_dict

        }

        if event_id in output_dict:
        output_dict[event_id].append(result_dict)
        else:
        output_dict[event_id] = [result_dict]

    output_json = {'phase2_classification_results': output_dict}

    return output_json

#Helper Fuctions
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

def load_megad_json(output_json):
    megad_json = {}

    with open(output_json) as json_file:
        data = json.load(json_file)

    data = data['phase2_classification_results']

    # key, value in image_id is in the format: '0': "SSWI000000020143548C.jpg"
    for key, value in data.items():
        for dict_list in value:
            key = dict_list['img_id']
            megad_json[key] = {}
            megad_json[key]['Count'] = len(dict_list['detections'])
            coord_list = ""
            conf_list = ""
            for bbox_conf in dict_list['detections']:
                # coords has a list of 4 elements
                bbox = ','.join([str(item) for item in bbox_conf['bbox']]) + ";"
                coord_list += bbox
                conf_list += str(bbox_conf['conf']) + ";"
            if (len(dict_list['detections']) == 0):
                megad_json[key]['Coords'] = ""
                megad_json[key]['Conf'] = ""
            else:
                megad_json[key]['Coords'] = coord_list[:-1]
                megad_json[key]['Conf'] = conf_list[:-1]

    # print(megad_json)
    return megad_json

def format_megad(megad_dict, model_id):
    '''Convert effnet dict to pandas df'''
    event_list = []
    image_list = []
    count_list = []
    bbox_list = []
    conf_list = []

    for key, value in megad_dict.items():
        event_list.append(img_name_to_event(key))
        image_list.append(key)
        count_list.append(value['Count'])
        bbox_list.append(value['Coords'])
        conf_list.append(value['Conf'])

    effnet_int_df = pd.DataFrame({'event':event_list,
                 'image':image_list,
                 'count': count_list,
                  'bbox':  bbox_list,
                 'conf': conf_list}).sort_values(by='image')

    image_group_id = []
    image_id_1 = []
    image_id_2 = []
    image_id_3 = []
    image_id_1_count = []
    image_id_2_count = []
    image_id_3_count = []
    image_id_1_bbox = []
    image_id_2_bbox = []
    image_id_3_bbox = []
    image_id_1_conf = []
    image_id_2_conf = []
    image_id_3_conf = []

    for event in effnet_int_df['event'].unique():
        image_group_id.append(event)
        event_effnet = effnet_int_df[effnet_int_df['event'] == event]

        i = 1
        for row, values in event_effnet.iterrows():
            image_appendix = values['image'].split('.')[0][-1]

            count_for_row = values['count']
            bbox_for_row = values['bbox']
            conf_for_row = values['conf']

            if i ==1:
                image_id_1.append(image_appendix)
                image_id_1_count.append(count_for_row)
                image_id_1_bbox.append(bbox_for_row)
                image_id_1_conf.append(conf_for_row)
            elif i == 2:
                image_id_2.append(image_appendix)
                image_id_2_count.append(count_for_row)
                image_id_2_bbox.append(bbox_for_row)
                image_id_2_conf.append(conf_for_row)
            elif i == 3:
                image_id_3.append(image_appendix)
                image_id_3_count.append(count_for_row)
                image_id_3_bbox.append(bbox_for_row)
                image_id_3_conf.append(conf_for_row)
            i+=1

        if i == 3:
            image_id_3.append('')
            image_id_3_count.append('')
            image_id_3_bbox.append('')
            image_id_3_conf.append('')
        if i == 2:
            image_id_2.append('')
            image_id_2_count.append('')
            image_id_2_bbox.append('')
            image_id_2_conf.append('')

            image_id_3.append('')
            image_id_3_count.append('')
            image_id_3_bbox.append('')
            image_id_3_conf.append('')

    formatted_megad = pd.DataFrame({'image_group_id':image_group_id,
        'image_id_1':image_id_1,
        'image_id_2':image_id_2,
        'image_id_3':image_id_3,
        'image_id_1_count': image_id_1_count,
        'image_id_2_count':image_id_2_count,
        'image_id_3_count':image_id_3_count,
        'image_id_1_conf': image_id_1_conf,
        'image_id_2_conf': image_id_2_conf,
        'image_id_3_conf': image_id_3_conf,
        'image_id_1_bbox': image_id_1_bbox,
        'image_id_2_bbox': image_id_2_bbox,
        'image_id_3_bbox': image_id_3_bbox})


    for image in range(1,4):
        formatted_megad['image_id_{}_species_name'.format(image)] = ''

    for image in range(1,4):
        formatted_megad['image_id_{}_blank'.format(image)] = formatted_megad['image_id_{}_count'.format(image)].apply\
                                                            (lambda x:True if x == 0 else False)

    for image in range(1,4):
        formatted_megad['image_id_{}_detectable'.format(image)] = False

    formatted_megad['model_id'] = model_id

    return formatted_megad


def run_format_megad(images_dir, json_path, model_id):

    print('Running Megadetector, Model ID {}'.format(model_id))
    run_md_inference(images_dir, json_path)

    output_json = reformat_md_json(json_path)
    stage2_megad_output_json = output_json

    stage2_megad_dict = load_megad_json(stage2_megad_output_json)
    formatted_megad = format_megad(stage2_megad_dict, model_id)

    return formatted_megad
