# capstone

To run YOLOv5 inference using species model weights:

````
python3 detect_yolo_animal.py --source "test/images" --weights "yolov5l_serengeti_swi_species_best.pt"
````

To run YOLOv5 inference using species model weights:

````
python3 detect_yolo_animal.py --source "test/images" --weights "yolov5s_serengeti_swi_blank_best.pt"
````

The detect_yolo_animal.py file (from src/backend folder) is expected to be placed in the same folder as "detect.py" under yolov5 folder. This file has `models` and `utils` dependencies (these 2 are folders with py code in yolov5 folder). The test/images folder and the model weights are expected to be in the same folder. If placed under separate folders, they can be referenced appropriately using the "--source" and "--weights" params
