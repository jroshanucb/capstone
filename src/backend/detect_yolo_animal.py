"""Run inference with a YOLOv5 model on images

Usage:
    $ python3 path/to/detect_yolo_animal.py --source path/to/img.jpg --weights path/to/model.pt --dbwrite='false' -modelid='3'
    example:
    python3 detect_yolo_animal.py --source "./../../data/test/yolo_splits3/test/images" --weights "../../../project/yolov5l_no_pretrain_swi_best.pt" --dbwrite='false' -modelid='3'

Author:
    Javed Roshan
    Modified the code from yolov5's detect.py
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import os

from db_conn import load_db_table
from db_conn import config
import pandas as pd
import psycopg2

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, colorstr, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_sync

model = None        # trained model to be loaded once
cmd_options = None  # command options to be used during inference

def modelLoad(
        weights='yolov5l_no_pretrain_swi_best.pt',  # model.pt path(s)
        source='test/images',  # folder to get the files from
        modelid='3',  # 1 = YOLOv5 blank model; 3 = YOLOv5 species model
        dbwrite='false', # flag that will write to DB
        conf_thres=0.25, # default=0.25
        iou_thres=0.45, # default=0.45
        imgsz=352,  # inference size (pixels)
        ):

    global model
    # Initialize
    set_logging()
    device = select_device('')

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # print('Image size is: ', imgsz, '; Stride: ', stride)

@torch.no_grad()
def run(filename, # include path of the file
        weights='yolov5l_no_pretrain_swi_best.pt',  # model.pt path(s)
        source='test/images',  # not relevant with MQTT
        modelid='3',  # 1 = YOLOv5 blank model; 3 = YOLOv5 species model
        dbwrite='false',
        conf_thres=0.25, # default=0.25
        iou_thres=0.45, # default=0.45
        imgsz=352,  # inference size (pixels)
        ):
    global model
    ret_msg = ''
    device = select_device('')

    # Read image
    path = source + "/" + filename
    img0 = cv2.imread(path)  # BGR
    im0s = img0
    assert img0 is not None, 'Image Not Found ' + path

    stride = int(model.stride.max())  # model stride

    # # Padded resize
    img = letterbox(img0, imgsz[0], stride=stride)[0]

    # Convert BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB
    img = np.ascontiguousarray(img)

    t0 = time.time()
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_sync()
    pred = model(img, augment=False, visualize=False)[0]

    # Apply NMS
    # pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    t2 = time_sync()

    # Process detections
    ret_preds = {} # predictions to be returned through this dictionary
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = path, '', im0s.copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # get the predictions to return
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                dict_class = ret_preds.get('Class', "empty")
                pred_class = str(cls.item())
                if (dict_class == "empty"):
                    ret_preds['Class'] = [pred_class]
                else:
                    ret_preds['Class'] = ret_preds['Class'] + [pred_class]
                dict_conf = ret_preds.get('Conf', "empty")
                pred_conf = str(conf.item())
                if (dict_conf == "empty"):
                    ret_preds['Conf'] = [pred_conf]
                else:
                    ret_preds['Conf'] = ret_preds['Conf'] + [pred_conf]
                dict_coords = ret_preds.get('Coords', "empty")
                pred_coords = ','.join(map(str,xywh))
                if (dict_coords == "empty"):
                    ret_preds['Coords'] = [pred_coords]
                else:
                    ret_preds['Coords'] = ret_preds['Coords'] + [pred_coords]

    return ret_preds

# Organize all files into a dictionary of events. This assumes filesnames have "eventId + fileId.jpg" structure
def organize_events(
        source='test/images/'
    ):
    # populate the dictionary to check which events have images
    imagesDict = {}
    count = 0
    for filename in os.listdir(source):
        image_name = filename.strip().split('.')[0]
        eventId = image_name[:-1]
        imageId = image_name[-1:]
        count = count + 1
        dict_eventId = imagesDict.get(eventId, "empty")
        if (dict_eventId == "empty"):
            imagesDict[eventId] = [imageId]
        else:
            imagesDict[eventId] = imagesDict[eventId] + [imageId]
    return imagesDict


def get_insert_stmt():
    sql_stmt = "insert into public.model_output ("
    sql_stmt += "model_ouput_id, model_id, image_group_id, "
    sql_stmt += "image_id_1, image_id_1_species_name, image_id_1_conf, image_id_1_count, image_id_1_blank, image_id_1_detectable, " 
    sql_stmt += "image_id_2, image_id_2_species_name, image_id_2_conf, image_id_2_count, image_id_2_blank, image_id_2_detectable, "
    sql_stmt += "image_id_3, image_id_3_species_name, image_id_3_conf, image_id_3_count, image_id_3_blank, image_id_3_detectable, "
    sql_stmt += "load_date) values "
    return sql_stmt

def get_speciesname_from_id(id):
    speciesList = ['bear', 'cottontail_snowshoehare', 'coyote', 'deer', 'elk', 'foxgray_foxred', 'opossum', 'raccoon', 'turkey', 'wolf']
    idx = int(id)
    if idx > 9 or idx < 0:
        speciesName = 'other'
    else:
        speciesName = speciesList[idx]
    return speciesName

def get_values_stmt(iteration, iter_size, modelid, model_output):
    sql_values_stmt = ""

    # Example model_output[key] = 
    # {'SSWI000000020365431C.jpg': {'Class': ['8.0', '8.0'], 
    #                                'Conf': ['0.7531381249427795', '0.8462810516357422'], 
    #                                'Coords': ['0.9179331064224243,0.7872340679168701,0.12158054858446121,0.09118541330099106', '0.38145896792411804,0.9088146090507507,0.2705167233943939,0.18237082660198212']
    #                               }, 
    #  'SSWI000000020365431B.jpg': {'Class': ['8.0', '8.0'], 
    #                                'Conf': ['0.7363986968994141', '0.7579455971717834'], 
    #                                'Coords': ['0.34802430868148804,0.9103343486785889,0.22796352207660675,0.17933130264282227', '0.8875380158424377,0.7857142686843872,0.13981762528419495,0.09422492235898972']
    #                               },
    #  'SSWI000000020365431A.jpg': {'Class': ['8.0', '8.0', '5.0'], 
    #                                'Conf': ['0.337464839220047', '0.41901835799217224', '0.4265201687812805'], 
    #                                'Coords': ['0.7644376754760742,0.7963525652885437,0.12462005764245987,0.08510638028383255', '0.1322188377380371,0.9194529056549072,0.2644376754760742,0.16109421849250793', '0.1322188377380371,0.9179331064224243,0.2583586573600769,0.16413374245166779']
    #                               }
    # }
    counter = 1
    for key, value in model_output.items():
        model_output_id = iteration * iter_size + counter
        counter = counter + 1
        image_group_id = key # this is the event_id
        sql_values_stmt += "(" + str(model_output_id) + ", " + modelid + ", '" + image_group_id + "', "
        for key2, value2 in value.items():
            dict1 = value2
            image_id = key2[-5:][0] # get 'C' from this file name 'SSWI000000020365431C.jpg'
            if (len(dict1.keys()) > 0): # for images where no species exist, the dict will be empty
                # ignore value2 for now
                # image_id_species_name = [get_speciesname_from_id(int(float(sn))) for sn in dict1['Class']]
                image_id_species_name = ','.join([get_speciesname_from_id(int(float(sn))) for sn in dict1['Class']])
                image_id_conf = ','.join([cf for cf in dict1['Conf']])
                image_id_count = len(dict1['Coords'])
                sql_values_stmt +=  "'" + image_id + "', '" + str(image_id_species_name) + "', '" + str(image_id_conf) + "', " + str(image_id_count)
                sql_values_stmt +=  ", false, false, "
            else:
                # empty image with no predictions
                sql_values_stmt +=  "'" + image_id + "', '', '', 0"
                sql_values_stmt +=  ", true, false, "

        load_date = "to_date('10-11-2021','DD-MM-YYYY')"
        sql_values_stmt += load_date + "), "

    return sql_values_stmt

def db_init():
    config_db = "database.ini"
    params = config(config_db)
    conn = psycopg2.connect(**params)

    return conn

def db_flush(iteration, iter_size, modelid, conn, model_output):
    # model_output has the format of 
    # model_output[image_group_id] = dict of fileInfer
    # fileInfer has the format of 
    # fileInfer[filename] = image_id, class (a number), coordinates (count from these numbers)

    sql_insert_stmt = get_insert_stmt()
    sql_values_stmt = get_values_stmt(iteration, iter_size, modelid, model_output)
    sql_stmt = sql_insert_stmt + sql_values_stmt[:-2]
    print("sql statment ---->", sql_stmt)
    cur = conn.cursor()
    cur.execute(sql_stmt)
    conn.commit()

    return

def process_images(
        weights='yolov5l_serengeti_swi_species_best.pt',  # model.pt path to the weights 
        source='test/images',  # path from where files have to be processed
        modelid='3',  # 1 = YOLOv5 blank model; 3 = YOLOv5 species model
        dbwrite='false', # flag that will write to DB
        conf_thres=0.25, # default=0.25
        iou_thres=0.45, # default=0.45
        imgsz=352
        ):
    global cmd_options

    iteration = 0
    # Organize events into a dictionary
    imagesDict = organize_events(source)
    conn = db_init()

    # for every event from the event list, perform yolo inference for all images from an event
    model_output = {}
    count = 0
    for key, value in imagesDict.items():
        fileInfer = {}
        for id in value:
            # filename = "SSWI000000006489319A.jpg"    #1 elk 
            # filename = "SSWI000000022151861A.jpg"    #2 bears
            filename = key+id+".jpg"

            # YOLO inference call
            ret_preds= run(filename, **vars(cmd_options))

            # "image_id_1, image_id_1_species_name, image_id_1_count, image_id_1_blank, image_id_1_detectable, "
            fileInfer[filename] = ret_preds
        model_output[key] = fileInfer
        print(fileInfer)
        count += 1

        # db flush for every 50 events
        if (dbwrite=='true' and count > 50):
            db_flush(iteration, 50, modelid, conn, model_output)
            iteration = iteration + 1
            model_output = {}
            count = 0
        elif count > 50:
            model_output = {}
            count = 0

    return

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='test/images/', help='path to get images for inference')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l_no_pretrain_swi_best.pt', help='best.pt path')
    parser.add_argument('--modelid', type=str, default='3', help='1 = YOLOv5 blank model; 3 = YOLOv5 species model')
    parser.add_argument('--dbwrite', type=str, default='false', help='db persistence enabler')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[352], help='inference size h,w')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    opt = parser.parse_args()
    return opt

def main(cmd_opts):
    global cmd_options
    cmd_options = cmd_opts
    modelLoad(**vars(cmd_options))
    process_images(**vars(cmd_options))

if __name__ == "__main__":
    cmd_opts = parse_opt()
    # print(cmd_opts)
    main(cmd_opts)