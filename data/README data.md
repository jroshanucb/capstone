# data

Notebooks used to select images for annotation work, bring images and labels back together in correct format and split images and labels into train/val/test.

__Yolo__

Pre-Processing

- **1. Resize SWI Images**: Change SWI expert labeled images to 329x329.
- **2. Parse Data for Annotation Work**: Pull out images into individual species folders for annotation work.
- **3. Annotations to Yolo Format**: Take our annotated labels and apply the correct labels. Move the labelled images into a single folder.

Data Splitting

- **4. Parse Yolo Data Splits**: Move annotated images and labels into train/val folders.

Test Image Processing

- **5. SWI test sampling**: Move images from SWI into test splits and write labels to csv.

Result Review

- **6. Yolo Val Result Review**: Take traditional results metrics and convert them to events.
- **7. Yolo Test Result Metrics**: Use inference txt file to determine test results by event