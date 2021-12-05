import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
import json
import cv2
from pathlib import Path
import glob
import re
import warnings
warnings.filterwarnings("ignore")

from model_inf.run_models import run_ensemble_models
from ensemble.full_ensemble import run_full_ensemble

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='models/test_images/', help='path to get images for inference')
    parser.add_argument('--truth', type=str, default='../data/test_labels.csv', help='path to get csv for true labels and counts\
    by event')
    parser.add_argument('--modelsz', type=str, default='small', help='model size: small, medium or large?')
    parser.add_argument('--dbwrite', type=str, default='false', help='db persistence enabler')
    parser.add_argument('--writeimages', type=str, default='true', help='write images with bbox')
    parser.add_argument('--imgsz',  type=int, default=329, help='inference size h,w (square)')


    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    cmd_opts = parse_opt()

    img_directory = cmd_opts.source
    truth_file_path = cmd_opts.truth
    modelsz = cmd_opts.modelsz
    dbwrite =  cmd_opts.dbwrite
    write_images =  cmd_opts.writeimages
    imgsz = cmd_opts.imgsz

    print(
    """\
============================================================
    Running CAMinos: Intelligent Trail Camera Annotation.

                      __.------~~~-.
                    ,'/             `\

                    " \  ,..__ | ,_   `\_,
                       >/|/   ~~\||`\(`~,~'
                       | `\     /'|   \_;
                       "   "   "
============================================================
    """
    )


    os.chdir('model_inf/')
    run_ensemble_models(img_directory = img_directory,
                    modelsz = modelsz,
                    dbwrite = dbwrite,
                    imgsz = imgsz)
    os.chdir('../')

    os.chdir('ensemble/')
    run_full_ensemble(modelsz = modelsz,
                    read_img_directory = img_directory,
                    truth_file_path = truth_file_path,
                    write_images = write_images,
                    img_size=imgsz)
