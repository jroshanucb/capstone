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

[**Python>=3.6.0**](https://www.python.org/), [**Pip3**](https://pip.pypa.io/en/stable/) is required with all
[requirements.sh](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/) and [**Tensorflow>=1.13**](https://github.com/tensorflow/tensorflow) or if running on GPU, [**Tensorflow-GPU>=1.13**](https://pypi.org/project/tensorflow-gpu/).
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
```

Arguments:

```python
--source 
	-path to get images for inference
	-default='../model_inf/test_images/'
--truth
	-path to get csv for true labels and counts by event
   	-default= '../data/test_labels.csv' 
--modelsz
    -model size: small, medium or large 
    -default='small' 	
--dbwrite
    -db persistence enabler'
    -default='false'  	
--writeimages
	-write images with bboxes
	-default='true'
--imgsz
	-inference image size h,w (square)
	-default=329
```
</details>

## <div align="center">Web Application</div>

<details open>
<summary><b>Run the Web App</b></summary>


To run the web application locally 

```bash
$ docker pull jroshanucb/capstone:latest
$ docker run --rm  -p 5000:5000 -p 8080:8080 jroshanucb/capstone:latest
```

Open a web browser & go to <b>http://localhost:8080/</b>

<b>What's in the Container?</b>

Repo [capstone-deploy](https://github.com/jroshanucb/capstone-deploy) has all the code that's in the container. 

Refer the [Dockerfle](https://github.com/jroshanucb/capstone-deploy/blob/main/src/backend/Dockerfile-capstone) for more details on  all the layers of the container.

</details>

<details open>
<summary><b>Web Environment Setup</b></summary>


Web development environment requires nodejs for npm. Using npm, we can install angular. 

<b>For Mac</b>
Install [nodejs version 12.16.1](https://nodejs.org/download/release/v12.16.1/)

```bash
$ sudo npm install @angular-devkit/build-angular@0.901.15
```

<b>For Ubuntu</b>
To install nodejs, following [article](https://www.tecmint.com/install-angular-cli-on-linux/) has the detailed steps. In summary, the steps are:

```bash
$ sudo curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash
$ sudo apt install -y nodejs
$ sudo npm install -g @angular/cli
```

Following steps remain the same irrespective of the environment for the rest of the setup:

Clone the [Capstone Repo](https://github.com/jroshanucb/capstone)

```bash
$ git clone https://github.com/jroshanucb/capstone.git
$ cd capstone/src/frontend
$ ng serve
```

Above command **ng serve**, makes the web application available at **http://localhost:4200**

To create a production package, run the following command:
```bash
$ ng build --prod
```

This generates the web deployable setup in the **dist** folder

[Nginx](https://www.nginx.com/) & Gunicorn [https://gunicorn.org/] are needed to deploy above generated code on a server or a container. They can be setup as follows:

```bash
$ sudo apt-get install python3-dev nginx
$ pip3 install gunicorn
```

<b>Nginx Configuration</b>

To start and stop Nginx, use following commands

```bash
$ sudo /etc/init.d/nginx start
$ sudo /etc/init.d/nginx stop
```

The deployment port of the angular prod build can be specified as follows:
```bash
    server {
        listen         8080 default_server;
        server_name    angular-deploy;
        root           /var/www/angular-deploy;
        index          index.html;
        try_files $uri /index.html;
    }
```
This configuration has to be added to the **/etc/nginx/nginx.conf** folder. A version of this file is included in the [repo here](https://github.com/jroshanucb/capstone-deploy/blob/main/src/backend/nginx.conf)

The contents of the [dist/newproject](https://github.com/jroshanucb/capstone-deploy/tree/main/src/frontend/dist/newproject) have to be deployed in the **/var/www/angular-deploy** folder as specified in the configuration

```bash
$ sudo cp -r /app/capstone-deploy/src/frontend/dist/newproject/* /var/www/angular-deploy/
```

After this deployment, ensure to stop and start the nginx server

<b>Gunicorn Configuration</b>

This python WSGI HTTP server, is bounded to the API server represented in the [code here](https://github.com/jroshanucb/capstone-deploy/blob/main/src/backend/app.py). Note that the same directory structure is enforced in the [main repo](https://github.com/jroshanucb/capstone/)

The wsgi code is [here](https://github.com/jroshanucb/capstone-deploy/blob/main/src/backend/wsgi.py)

```bash
$ gunicorn --bind 0.0.0.0:5000 wsgi:app
```

The above command makes the API backend server run on port 5000. This is the url:port that frontend code refers in the [API Service Type Script Code](https://github.com/jroshanucb/capstone/blob/main/src/frontend/src/app/apiservice.service.ts)

</details>

<details open>
<summary><b>What's needed to deploy this Web App on a different server?</b></summary>


To deploy the webserver on a different server environment, ensure appropriate setups from **Web Environment Setup** section are completed.

<b>Change the API Service code</b>

The [API Service Type Script Code](https://github.com/jroshanucb/capstone/blob/main/src/frontend/src/app/apiservice.service.ts) has to be modified to reflect the new IP. Also ensure that the IP:port are available from public internet if this web application has to be accessible to others. On an AWS setup, ensure the security group has appropriate traffic enabled on the port 5000. Also ensure that a static IP (or a dns/host-name) is assined/linked to the EC2 server on which the web app is deployed.

<b>Rebuild the code</b>

```bash
$ ng build --prod
```

<b>Update the nginx setup</b>

The output from the **dist/newproject** shall be copied to the **/var/www/angular-deploy** folder

```bash
$ sudo cp -r /app/capstone-deploy/src/frontend/dist/newproject/* /var/www/angular-deploy/
```

<b>Restart the nginx server</b>

```bash
$ sudo /etc/init.d/nginx start
$ sudo /etc/init.d/nginx stop
```

Access the web application from the browser using the url specified in the API Service Type Script Code. 

Note that the API service and the webserver are running on the same EC2 instance in this configuration but they are running on 2 different ports (5000 & 8080 respectively)

</details>

# Repo Structure

    ├── README.md	<- The top-level README for developers using this project.
    ├── caminos_inference.py	<- Script to run full inference as specified above.
    ├── download_test_image_set.sh	<- download test data (6k images from SWI)
    ├── requirements.sh  <- download and install packages required to run full caminos inference
    ├── data    <- directory containing series of .pynb notebooks to convert images to required format and structure needed to run training. 
    ├── model_inf    <- directory containing scripts to run inference on images. These are all run through caminos_inference.py.
    ├── ensemble    <- directory containing scripts merge inference results through ensemble logic. These are all run through caminos_inference.py.
    ├── results    <- directory containing results files after running inference, including final predictions and images with bounding boxes. These are all produced through caminos_inference.py.
    ├── src    <- front end.
    ├── train    <- pynb notebook to train efficientnet.
    ├── extras    <- misc images
