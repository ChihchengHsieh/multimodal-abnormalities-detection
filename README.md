# multimodal-abnormalities-detection

In this work, we re-designed a multimodal architecture that inspired by Mask RCNN. And, our porposed model can receive two modalities as input, including tabular data and image data. To reason to create this multimodal architecture is to prove a theory that the clinical data (age, gender, blood pressure, etc.) can improve the peformance of abnormality detecton. The traditional approach use only Chest X-ray image to detect lesions. However, according to our radiologists, they will need clinical data to provide a more precise diagnosis. Therefore, we decide to test if it's the case for machine learninig model. In order to let the AI model process two modalities (clincal data & CXR image) at the same time, we then studied and attempted several fusion and feature engineering methods to finally developed this architecture. 

Because the architecture is inspired by Mask RCNN, it use almost the same strategy to process image feature. The image below shows the model architecture when no clinical data is used.

![NoClinicalArch](https://github.com/ChihchengHsieh/multimodal-abnormalities-detection/blob/master/charts/FasterRCNN-MobileNet.png?raw=true)

And this is the architecture we proposed to cope with image and tabular data at the same time.

![WithClinicalArch](https://github.com/ChihchengHsieh/multimodal-abnormalities-detection/blob/master/charts/Multimodal%20Faster%20R-CNN%20MobileNet.png?raw=true)

To make the architecture more understandable, the we use the MobileNet as the backebone of the detector network, and the Feature Pyramid Network (FPN) is mentioned in the figures above. However, the MobileNet can be replaced by any others networks that can process image data. And, the FPN are often used to improve the model performance. 

In this task, we retrieve the bounidng boxes from *REFLACX* dataset. However, this dataset actually provide the bouding ellipses but not bounding boxes. We simply transform their bounding ellipses to boxes that containg those ellipses. The segmentation lable is also transformed form the bounding boxes. The reason using Mask RCNN rather than Faster RCNN is that we found the peformance slightly improve by adding the segmentation head in the output. And, this is the case in [this work](https://www.sciencedirect.com/science/article/abs/pii/S0263224119305202).



# Dealing with overfitting.

As we replace the resnet50 by resnet18, the training graph showing this.

<img width="603" alt="image" src="https://user-images.githubusercontent.com/37566901/164976916-dad2472c-dd4c-4e6c-955d-59643dc0e19c.png">


As the figure shown, it started showing the overfitting at around 235th epoch. From here, our next move is to increase `learning rate` (to boost the training speed, it's slow and took more than 10 hours to converge) and `weight_decay` (see if the model remain trainable).

<img width="602" alt="image" src="https://user-images.githubusercontent.com/37566901/164991636-b48100b7-3c53-4b94-9697-38029289599b.png">
And the figure above is the result using larger `learning_rate` and `weight_decay`.

```
[model]: 36,233,555
[model.backbone]: 3,602,468
[model.rpn]: 2,435,595
[model.roi_heads]: 30,195,492
[model.roi_heads.mask_head]: 2,959,360
[model.roi_heads.box_head]: 26,941,440 # most of the parameters living inside here.
[model.roi_heads.box_head.fc6]: 25,891,840
[model.roi_heads.box_head.fc7]: 1,049,600
[model.roi_heads.box_predictor]: 30,750
```

And I found that most of the parameters living inside the model is in the `model.roi_heads.box_head`, which is constructed with 

```python
box_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
```

And, the `out_channels` is the channel size of image NN output. 

The next experiment we do is to shrink the model `out_channels` to `16`.

However, the overfitting still remained.

<img width="599" alt="image" src="https://user-images.githubusercontent.com/37566901/165051475-e241f8cd-5c71-439f-b62d-25ea3887f620.png">

so we tried another setting, which gave us this model:

```
[model]: 3,434,031
[model.backbone]: 1,092,928
[model.rpn]: 11,723
[model.roi_heads]: 2,329,380
[model.roi_heads.mask_head]: 1,844,224
[model.roi_heads.box_head]: 217,344
[model.roi_heads.box_head.fc6]: 200,832
[model.roi_heads.box_head.fc7]: 16,512
[model.roi_heads.box_predictor]: 3,870

ModelSetup(
    name="overfitting_1",
    use_clinical=False,
    use_custom_model=True,
    use_early_stop_model=True,
    backbone="mobilenet_v3",
    optimiser="sgd",
    lr=1e-2,
    pretrained=True,
    dataset_mode="unified",
    image_size=256,
    weight_decay=5e-3,
    record_training_performance=True,
    using_fpn=False,
    backbone_out_channels=32, # shrink size test [32]
    representation_size=128, # shrink size test [128]
)
```
<img width="594" alt="image" src="https://user-images.githubusercontent.com/37566901/165109242-e41dd0d4-674e-46fa-a1ad-3c90fd352826.png">

And another,

```
[model]: 3,141,215
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 2,127,652
[model.roi_heads.mask_head]: 1,807,360
[model.roi_heads.box_head]: 54,400
[model.roi_heads.box_head.fc6]: 50,240
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
```
