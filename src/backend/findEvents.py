import argparse
import os

imagesDict = {}

def process_images(
        source='../../../data/test/yolo_splits3/test/images/'
    ):
    # populate the dictionary to check which events have images
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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='../../../data/test/yolo_splits3/test/images/', help='path to get images for inference')
    opt = parser.parse_args()
    return opt

def main(cmd_opts):
    process_images(**vars(cmd_opts))
    print("Total Events: ", len(imagesDict.keys()))
    for key, value in imagesDict.items():
        if len(value) < 3:
            print("Event: ", key, " , Images: ", value)

if __name__ == "__main__":
    cmd_opts = parse_opt()
    main(cmd_opts)