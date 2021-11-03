"""Run inference with a YOLOv5 model on images

Usage:
    $ python3 path/to/detect_yolo_animal.py --source path/to/img.jpg --weights path/to/model.pt

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
import torch.backends.cudnn as cudnn
import numpy as np
import os


FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, colorstr, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized

model = None        # trained model to be loaded once
cmd_options = None  # command options to be used in the MQTT message loop

def modelLoad(
        weights='yolov5l_serengeti_swi_species_best.pt',  # model.pt path(s)
        source='test/images',  # not relevant with MQTT
        imgsz=640,  # inference size (pixels)
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
        weights='yolov5l_serengeti_swi_species_best.pt',  # model.pt path(s)
        source='test/images',  # not relevant with MQTT
        imgsz=640,  # inference size (pixels)
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
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    t0 = time.time()
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False, visualize=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    t2 = time_synchronized()

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

    return ret_class, ret_msg


def process_images(
        weights='yolov5l_serengeti_swi_species_best.pt',  # model.pt path to the weights 
        source='test/images',  # path from where files have to be processed
        ):
    global cmd_options

    count = 0
    for filename in os.listdir(source):
        # filename = "SSWI000000006489319A.jpg"    #1 elk 
        # filename = "SSWI000000022151861A.jpg"    #2 bears

        # YOLO inference call
        ret_class, coords = run(filename, **vars(cmd_options))

        model_output_msg = "Filename: {}; {}; Bbox[list]: {}".format(filename, ret_class, coords)
        print(model_output_msg)
        
        count = count + 1
        # if (count > 10):
        #     break

    return

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='test/images/', help='path to get images for inference')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5l_serengeti_swi_species_best.pt', help='best.pt path')
    opt = parser.parse_args()
    return opt

def main(cmd_opts):
    global local_mqttclient
    global cmd_options
    cmd_options = cmd_opts
    # Loop listener forever
    modelLoad(**vars(cmd_options))
    process_images(**vars(cmd_options))

if __name__ == "__main__":
    # update cmd_options global variable
    cmd_opts = parse_opt()
    main(cmd_opts)