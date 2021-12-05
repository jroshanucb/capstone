# CAMinos: Intelligent Trail Camera Annotation

<br>
<p>
CAMinos 🦌 applies computer vision to trail camera images to help automate the annotation process. Our deep learning system predicts both the animal species and the count of animals caught on trail camera images using an ensemble approach that consists of both object detection and classification algorithms. These predictions are then supplied as recommendations to users as they label images in our custom-built annotation tool. Caminos streamlines the annotation process for users, resulting in much faster annotation times and providing a valuable resource to the wildlife conservation community. 

</p>

<!--
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->


## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.ssh](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/jroshanucb/capstone
$ cd capstone
```

<b>Install Requirements</b>

Ubuntu

```bash
$ chmod +x ./requirements.sh
$ ./requirements.sh
```

Mac

```bash
$ sh requirements.sh
```
Windows

```bash
$ bash requirements.sh
```

</details>

<details open>
<summary><b>Inference</b></summary>

Model inference can be run in 3 different sizes:

- <u>Small:</u> 
	- Efficientnet b0 Blank 
	- Yolov5 Species and Count
- <u>Medium:</u> 
	- Efficientnet b0 Blank
	- Yolov5 Blank
	- Yolov5 Species and Count
	- Efficientnet b5 Species
- <u>Large:</u>
	- Efficientnet b0 Blank
	- Yolov5 Blank
	- Yolov5 Species and Count
	- Efficientnet b5 Species
	- Megadetector Count

Example

```python
python caminos_inference.py --source ../model_inf/test_images/

Arguments:
	--source 
		-path to get images for inference
		-default='../model_inf/test_images/'
	
	--truth
		-path to get csv for true labels and counts
    by event, 
   	 	-default= '../data/test_labels.csv'
    
    --modelsz
    	-model size: small, medium or large 		-default='small'
    	
    --dbwrite
    	-db persistence enabler'
    	-default='false'
    	
	--writeimages
		-write images with bboxes
		-default='true'
	--imgsz
		-inference image size h,w (square)
		-default=329
