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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    t0 = time.time()
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)[0]

    # print("{}, {}".format(type(pred), pred.shape))

    # top5 predictions using Softmax
    sm = nn.Softmax(dim=1)
    probabilities = sm(pred)

    _ , i = torch.topk(pred, 5)
    prob, _ = probabilities.topk(5)
    
    # for _ in range(1):
    #     print(i.shape)

    dict_preds = {}
    itr = 0
    print(i.cpu().numpy()[-1].shape)
    # for x in i.cpu().numpy()[-1]:

    #     if x in dict_preds:
    #         dict_preds[int(x)].append(float(i.cpu().detach().numpy()[0][itr]))
    #     else:
    #         dict_preds[int(x)] = [float(i.cpu().detach().numpy()[0][itr])]
    #     itr += 1
        
    # print(dict_preds)

    # Apply NMS
    # pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    t2 = time_sync()

    # Process detections
    ret_class = ''
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = path, '', im0s.copy()

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # get the predictions to return
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                ret_class += "Class:" + str(cls.item()) + ","
                ret_class += "Conf:" + str(conf.item()) + ";"
                line = ','.join(map(str,xywh)) + ';'
                ret_msg += str(line)

    return ret_class, ret_msg, dict_preds

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
    sql_stmt += "image_id_1, image_id_1_species_name, image_id_1_count, image_id_1_blank, image_id_1_detectable, " 
    sql_stmt += "image_id_2, image_id_2_species_name, image_id_2_count, image_id_2_blank, image_id_2_detectable, "
    sql_stmt += "image_id_3, image_id_3_species_name, image_id_3_count, image_id_3_blank, image_id_3_detectable, "
    sql_stmt += "load_date) values "

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

    # {'SSWI000000017053464A.jpg': 
    # 'A;Class:5.0,Conf:0.2643603980541229;Class:8.0,Conf:0.7807839512825012;
    # 0.6778115630149841,0.630699098110199,0.054711245000362396,0.11854103207588196;
    # 0.2796352505683899,0.6823708415031433,0.15197569131851196,0.11246200650930405;}
    counter = 1
    for key, value in model_output.items():
        model_output_id = iteration * iter_size + counter
        counter = counter + 1
        image_group_id = key # this is the event_id
        sql_values_stmt += "(" + str(model_output_id) + ", " + modelid + ", '" + image_group_id + "', "
        for key2, value2 in value.items():
            valueList = value2.strip().split(';') #should return id, ret_class, coords
            sql_values_stmt += valueList[0] + ", '" + get_speciesname_from_id(valueList[1]) + "', "

        load_date = "to_date('10-11-2021','DD-MM-YYYY')"
        sql_values_stmt += load_date + "), "

    return sql_values_stmt


def db_flush(iteration, iter_size, modelid, model_output):
    # model_output has the format of 
    # model_output[image_group_id] = dict of fileInfer
    # fileInfer has the format of 
    # fileInfer[filename] = image_id, class (a number), coordinates (count from these numbers)

    config_db = "database.ini"
    params = config(config_db)
    conn = psycopg2.connect(**params)

    sql_insert_stmt = get_insert_stmt()
    sql_values_stmt = get_values_stmt(iteration, iter_size, model_output)
    sql_stmt = sql_insert_stmt + sql_values_stmt[:-2]
    # print(sql_stmt)
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

    # for every event from the event list, perform yolo inference for all images from an event
    model_output = {}
    count = 0
    for key, value in imagesDict.items():
    # for filename in os.listdir(source):
        fileInfer = {}
        for id in value:
            # filename = "SSWI000000006489319A.jpg"    #1 elk 
            # filename = "SSWI000000022151861A.jpg"    #2 bears
            filename = key+id+".jpg"

            # YOLO inference call
            ret_class, coords, dict_preds = run(filename, **vars(cmd_options))

            # "image_id_1, image_id_1_species_name, image_id_1_count, image_id_1_blank, image_id_1_detectable, "
            fileInfer[filename] = "{};{}{}".format(id, ret_class, coords)
            print(fileInfer)
        model_output[key] = fileInfer
        count += 1

        # db flush for every 100 events
        if (dbwrite=='true' and count > 100):
            db_flush(iteration, 100, modelid, model_output)
            iteration = iteration + 1
            model_output = []
        elif count > 100:
            model_output = []

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