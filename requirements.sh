#Requirements

#Download weights
cd model_inf

wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies  'https://docs.google.com/uc?export=download&id=19d-h3EH1YyMwfwWk9705TPd1iNvFpKPO' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=19d-h3EH1YyMwfwWk9705TPd1iNvFpKPO" -O weights.zip && rm -rf /tmp/cookies.txt

#unzip weights

unzip weights.zip

#Clone Directories for megadetector
git clone https://github.com/microsoft/CameraTraps/
git clone https://github.com/microsoft/ai4eutils/

#Set python virtual env for megadetector packages
cd ..
export PYTHONPATH="$PYTHONPATH:$PWD/model_inf/ai4eutils:$PWD/model_inf/CameraTraps"
export PYTHONPATH="$PYTHONPATH:$PWD/model_inf:$PWD/model_inf"

#install pip 
sudo apt install python3-pip

# If you have a GPU on your computer, change "tensorflow" to "tensorflow-gpu"
#pip install tensorflow==1.13.1
#pip install tensorflow-gpu==1.13.1

# Install other dependencies
pip install efficientnet_pytorch
pip install timm
pip install pandas tqdm pillow humanfriendly matplotlib tqdm jsonpickle statistics requests
pip install sklearn
pip install opencv-python
pip install pathlib
