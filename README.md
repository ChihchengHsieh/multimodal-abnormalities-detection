# multimodal-abnormalities-detection

In this work, we re-designed a multimodal architecture that inspired by Mask RCNN. And, our porposed model can receive two modalities as input, including tabular data and image data. To reason to create this multimodal architecture is to prove a theory that the clinical data (age, gender, blood pressure, etc.) can improve the peformance of abnormality detecton. The traditional approach use only Chest X-ray image to detect lesions. However, according to our radiologists, they will need clinical data to provide a more precise diagnosis. Therefore, we decide to test if it's the case for machine learninig model. In order to let the AI model process two modalities (clincal data & CXR image) at the same time, we then studied and attempted several fusion and feature engineering methods to finally developed this architecture. 

Because the architecture is inspired by Mask RCNN, it use almost the same strategy to process image feature. The image below shows the model architecture when no clinical data is used.

![NoClinicalArch](https://github.com/ChihchengHsieh/multimodal-abnormalities-detection/blob/master/charts/FasterRCNN-MobileNet.png?raw=true)

And this is the architecture we proposed to cope with image and tabular data at the same time.

![WithClinicalArch](https://github.com/ChihchengHsieh/multimodal-abnormalities-detection/blob/master/charts/Multimodal%20Faster%20R-CNN%20MobileNet.png?raw=true)

