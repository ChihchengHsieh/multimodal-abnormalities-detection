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
    weight_decay=1e-3,
    record_training_performance=True,
    using_fpn=False,
    backbone_out_channels=16,  # shrink size test [32]
    representation_size=128,  # shrink size test [128]
    mask_hidden_layers=256,
    use_mask=False,
)
    
[model]: 1,134,425
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 120,862
[model.roi_heads.box_head]: 116,992
[model.roi_heads.box_head.fc6]: 100,480
[model.roi_heads.box_head.fc7]: 16,512
[model.roi_heads.box_predictor]: 3,870
   
```
<img width="597" alt="image" src="https://user-images.githubusercontent.com/37566901/165158376-d92dfa88-34da-46f9-ad8d-d5f79f73cf16.png">

```
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
        weight_decay=1e-3,
        record_training_performance=True,
        using_fpn=False,
        backbone_out_channels=16,  # shrink size test [32]
        representation_size=64,  # shrink size test [128]
        mask_hidden_layers=256,
        use_mask=False,
    )
```

<img width="602" alt="image" src="https://user-images.githubusercontent.com/37566901/165170612-c272868a-fcd6-41c9-8929-caf74c7d7e96.png">


## Using a larger lr

```
    ModelSetup(
        name="overfitting_1",
        use_clinical=False,
        use_custom_model=True,
        use_early_stop_model=True,
        backbone="mobilenet_v3",
        optimiser="sgd",
        lr=5e-2,
        pretrained=True,
        dataset_mode="unified",
        image_size=256,
        weight_decay=1e-3,
        record_training_performance=True,
        using_fpn=False,
        backbone_out_channels=16,  # shrink size test [32]
        representation_size=64,  # shrink size test [128]
        mask_hidden_layers=256,
        use_mask=False,
    ),
    
```

### Shrink representation size:


```
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
        weight_decay=1e-3,
        record_training_performance=True,
        using_fpn=False,
        backbone_out_channels=16,  # shrink size test [32]
        representation_size=32,  # shrink size test [128]
        # mask_hidden_layers=64,
        use_mask=False,
    )
```

This one can try

###  custom  model (smaller)

```
[model]: 197,497
[model.backbone]: 166,736
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990

ModelSetup(
        name="overfitting_1",
        use_clinical=False,
        use_custom_model=True,
        use_early_stop_model=True,
        backbone="custom", # [mobilenet_v3]
        optimiser="sgd",
        lr=1e-2,
        pretrained=True,
        dataset_mode="unified",
        image_size=256,
        weight_decay=1e-3,
        record_training_performance=True,
        using_fpn=False,
        backbone_out_channels=16,  # shrink size test [32]
        representation_size=32,  # shrink size test [128, 64]
        # mask_hidden_layers=64,
        use_mask=False,
    )
```

### samll 

```
[model]: 732,281
[model.backbone]: 701,520
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```

### MobileNetV3 with samll batch size, according to [this work](https://papers.nips.cc/paper/2019/hash/dc6a70712a252123c40d2adba6a11d84-Abstract.html), the value of `lr/batch_size` should be biggere to reach a better generalization performance.

```
model_setup = ModelSetup(
    name="ov_3",
    use_clinical=False,
    use_custom_model=True,
    use_early_stop_model=True,
    backbone="mobilenet_v3",  # [mobilenet_v3]
    optimiser="sgd",
    lr=1e-2,
    pretrained=True,
    dataset_mode="unified",
    image_size=256,
    weight_decay=1e-3,
    record_training_performance=True,
    using_fpn=False,
    backbone_out_channels=16,  # shrink size test [32]
    representation_size=32,  # shrink size test [128, 64]
    # mask_hidden_layers=64,
    use_mask=False,
    batch_size=4,
)

[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```

<img width="592" alt="image" src="https://user-images.githubusercontent.com/37566901/165334184-1a4f77bc-6f7f-4629-b6a3-258b0397b2a1.png">
Still overfitting

## Then we try apply two layers of dropout in `XAMITwoMLPHead` but using the same model_setup.

<img width="596" alt="image" src="https://user-images.githubusercontent.com/37566901/165358982-b2acba40-d8b2-4f05-8c2a-09e9019a3f06.png">
start seeing the overfitting at around 60th epoch, the training decided to move on while the validation set still struggling around 0.25 AP. 

## Further increase the L2 (weight_decay=1e-1).
so we want to improve till around 20th epoch, and freeze their weight so on.

```
model_setup = ModelSetup(
    name="ov_3",
    use_clinical=False,
    use_custom_model=True,
    use_early_stop_model=True,
    backbone="mobilenet_v3",  # [mobilenet_v3]
    optimiser="sgd",
    lr=1e-3, # decrase lr cuz 1e-2 is not trainable.
    pretrained=True,
    dataset_mode="unified",
    image_size=256,
    weight_decay=1e-1,
    record_training_performance=True,
    using_fpn=False,
    backbone_out_channels=16,  # shrink size test [32]
    representation_size=32,  # shrink size test [128, 64]
    # mask_hidden_layers=64,
    use_mask=False,
    batch_size=4,
)
```


### With dropout=0.3 in box_head.
<img width="398" alt="image" src="https://user-images.githubusercontent.com/37566901/165445268-14a727ef-751d-42e4-8638-38e9ea2b71ad.png">


### With dropout=0.2 in box_head, StepLR.

<img width="400" alt="image" src="https://user-images.githubusercontent.com/37566901/165483874-b8fce143-f2de-411f-897f-93da96664e07.png">


### dropout=0.2, with StepLR.

<img width="406" alt="image" src="https://user-images.githubusercontent.com/37566901/165579785-c8e47609-28eb-4654-a26f-6b1c331f8405.png">

### dropout=0, with ReduceLROnPlateau.

<img width="400" alt="image" src="https://user-images.githubusercontent.com/37566901/165593268-340525a2-5525-4b2d-a3ab-70967d30c8b9.png">

### dropout=0, with ReduceLROnPlateau. patience = 5, factor = 0.5

<img width="402" alt="image" src="https://user-images.githubusercontent.com/37566901/165620898-7946c6fd-242e-4180-96ff-3d80a02e5c58.png">

### dropout=0, with ReduceLROnPlateau. patience = 3, factor = 0.5
```
val_ar_0_4523_ap_0_2463_test_ar_0_4995_ap_0_2929_epoch61_WithoutClincal_04-28-2022 07-36-46_ov_3
```
<img width="402" alt="image" src="https://user-images.githubusercontent.com/37566901/165635523-4fdb0be3-f8be-428c-9698-565bf334a1f2.png">

### dropout=0.2, with ReduceLROnPlateau. patience = 2, factor = 0.5

```
val_ar_0_3721_ap_0_2119_test_ar_0_3481_ap_0_2607_epoch74_WithoutClincal_04-28-2022 09-59-40_ov_3
```
<img width="591" alt="image" src="https://user-images.githubusercontent.com/37566901/165653321-e359b8e1-c2b5-4dfd-9f2f-64b6dd637f52.png">



# Model without clinical

### dropout=0, with ReduceLROnPlateau, factor=0.1, patience=3 (this is the baseline with pretty decent result) (v=0.3130, t=0.3557, best_v=0.3597, best_t=0.3714)

```
# ModelSetup(
#     name="ov_1",
#     use_clinical=False,
#     use_custom_model=True,
#     use_early_stop_model=True,
#     backbone="mobilenet_v3",  # [mobilenet_v3]
#     optimiser="sgd",
#     lr=1e-2,
#     pretrained=True,
#     dataset_mode="unified",
#     image_size=256,
#     weight_decay=1e-3,
#     record_training_performance=True,
#     using_fpn=False,
#     backbone_out_channels=16,  # shrink size test [16, 32]
#     representation_size=32,  # shrink size test [32, 64, 128]
#     # mask_hidden_layers=64,
#     use_mask=False,
#     batch_size=4,
#     box_head_dropout_rate= 0 , # [0, 0.1, 0.2, 0.3]
#     warmup_epochs=0,
#     lr_scheduler = "ReduceLROnPlateau", # [ReduceLROnPlateau, MultiStepLR]
#     reduceLROnPlateau_factor = 0.1,
#     reduceLROnPlateau_patience = 3,
#     multiStepLR_milestones= [30, 50, 70, 90] ,
#     multiStepLR_gamma =0.1,
# )

[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990

========================================For Training [ov_3]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='ov_3', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, using_fpn=False, representation_size=32, mask_hidden_layers=256, use_mask=False, batch_size=4, box_head_dropout_rate=0, warmup_epochs=0)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_6132_ap_0_3597_test_ar_0_5281_ap_0_3714_epoch26_WithoutClincal_04-28-2022 12-05-57_ov_3]
Best AR validation model has been saved to: [val_ar_0_7046_ap_0_3173_test_ar_0_6297_ap_0_3477_epoch19_WithoutClincal_04-28-2022 11-55-16_ov_3]
The final model has been saved to: [val_ar_0_5581_ap_0_3130_test_ar_0_5281_ap_0_3557_epoch100_WithoutClincal_04-28-2022 13-58-50_ov_3]

===================================================================================================
```
<img width="593" alt="image" src="https://user-images.githubusercontent.com/37566901/165677253-a28e2754-1613-4819-902f-64a774aad205.png">

### dropout=0, with MultiStepLR, multiStepLR_milestones=[30, 50, 70, 90] (v=0.2944, t=0.3029, best_v=0.3713, best_t=0.2908)

```
========================================For Training [ov_1]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='ov_1', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, using_fpn=False, representation_size=32, mask_hidden_layers=256, use_mask=False, batch_size=4, box_head_dropout_rate=0, warmup_epochs=0, lr_scheduler='MultiStepLR', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_6189_ap_0_3713_test_ar_0_5672_ap_0_2908_epoch13_WithoutClincal_04-28-2022 23-24-36_ov_1]
Best AR validation model has been saved to: [val_ar_0_6874_ap_0_2998_test_ar_0_6156_ap_0_3008_epoch31_WithoutClincal_04-28-2022 23-52-33_ov_1]
The final model has been saved to: [val_ar_0_4643_ap_0_2944_test_ar_0_5245_ap_0_3029_epoch100_WithoutClincal_04-29-2022 01-39-11_ov_1]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```
<img width="599" alt="image" src="https://user-images.githubusercontent.com/37566901/165864330-725ac9d4-bf37-4842-a6fd-6ef37d823ff2.png">

### dropout=0, with ReduceLROnPlateau, factor=0.1, patience=3, slightly larger model. (v=0.2996, t=0.2971, best_v=0.3464, best_t=0.3487)

```
========================================For Training [ov_2]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='ov_2', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=32, using_fpn=False, representation_size=64, mask_hidden_layers=256, use_mask=False, batch_size=4, box_head_dropout_rate=0, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_6729_ap_0_3464_test_ar_0_6509_ap_0_3487_epoch13_WithoutClincal_04-29-2022 02-03-01_ov_2]
Best AR validation model has been saved to: [val_ar_0_6729_ap_0_3464_test_ar_0_6509_ap_0_3487_epoch13_WithoutClincal_04-29-2022 02-03-01_ov_2]
The final model has been saved to: [val_ar_0_5735_ap_0_2996_test_ar_0_4763_ap_0_2971_epoch100_WithoutClincal_04-29-2022 04-14-58_ov_2]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 1,211,177
[model.backbone]: 1,092,928
[model.rpn]: 11,723
[model.roi_heads]: 106,526
[model.roi_heads.box_head]: 104,576
[model.roi_heads.box_head.fc6]: 100,416
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
```
<img width="612" alt="image" src="https://user-images.githubusercontent.com/37566901/165864493-4b461959-9adc-4548-9200-577b60b798eb.png">

### dropout=0.2, with ReduceLROnPlateau, factor=0.1, patience=3 (v=0.0645, t=0.0751, best_v=0.0698, best_t=0.0830) (it should have some warmup epochs that I forgot to set.)
```
========================================For Training [ov_3]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='ov_3', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, using_fpn=False, representation_size=32, mask_hidden_layers=256, use_mask=False, batch_size=4, box_head_dropout_rate=0.2, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_1202_ap_0_0698_test_ar_0_1158_ap_0_0830_epoch13_WithoutClincal_04-29-2022 04-38-10_ov_3]
Best AR validation model has been saved to: [val_ar_0_2725_ap_0_0595_test_ar_0_2289_ap_0_0456_epoch1_WithoutClincal_04-29-2022 04-19-05_ov_3]
The final model has been saved to: [val_ar_0_0909_ap_0_0645_test_ar_0_1158_ap_0_0751_epoch100_WithoutClincal_04-29-2022 06-51-09_ov_3]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```
<img width="604" alt="image" src="https://user-images.githubusercontent.com/37566901/165864744-76f2758b-99e0-41f4-b087-0babdde5d77a.png">


### dropout=0, with ReduceLROnPlateau, factor=0.5, patience=2 (v=0.3257, t=0.3381, best_v=0.3580, best_t=0.2964)
```
========================================For Training [ov_4]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='ov_4', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, using_fpn=False, representation_size=32, mask_hidden_layers=256, use_mask=False, batch_size=4, box_head_dropout_rate=0, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.5, reduceLROnPlateau_patience=2, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_5020_ap_0_3580_test_ar_0_4743_ap_0_2964_epoch12_WithoutClincal_04-29-2022 07-13-00_ov_4]
Best AR validation model has been saved to: [val_ar_0_6060_ap_0_3205_test_ar_0_5386_ap_0_2926_epoch18_WithoutClincal_04-29-2022 07-22-28_ov_4]
The final model has been saved to: [val_ar_0_4878_ap_0_3257_test_ar_0_4904_ap_0_3381_epoch100_WithoutClincal_04-29-2022 09-28-21_ov_4]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```
<img width="599" alt="image" src="https://user-images.githubusercontent.com/37566901/165864705-ec13f018-46b4-4c8a-ac74-b59fa7614279.png">


### dropout=0, with MultiStepLR, multiStepLR_milestones=[10 , 30, 50, 70, 90], multiStepLR_gamma=0.1 (v=0.3469, t=0.2877, best_v=0.3696, best_t=0.3018)

```
========================================For Training [ov_5]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='ov_5', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, using_fpn=False, representation_size=32, mask_hidden_layers=256, use_mask=False, batch_size=4, box_head_dropout_rate=0, warmup_epochs=0, lr_scheduler='MultiStepLR', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[10, 30, 50, 70, 90], multiStepLR_gamma=0.1)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_6674_ap_0_3696_test_ar_0_5029_ap_0_3018_epoch21_WithoutClincal_04-29-2022 11-34-45_ov_5]
Best AR validation model has been saved to: [val_ar_0_6681_ap_0_3612_test_ar_0_5634_ap_0_3244_epoch29_WithoutClincal_04-29-2022 11-47-28_ov_5]
The final model has been saved to: [val_ar_0_6103_ap_0_3469_test_ar_0_5279_ap_0_2877_epoch100_WithoutClincal_04-29-2022 13-38-12_ov_5]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```
<img width="601" alt="image" src="https://user-images.githubusercontent.com/37566901/165916489-c5777948-def8-4fe8-98dc-78627d5f9c4f.png">


### dropout=0, with ReduceLROnPlateau, factor=0.1, patience=2 (v=0.3210, t=0.3096, best_v=0.3810, best_t=0.3387)
```
========================================For Training [ov_6]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='ov_6', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, using_fpn=False, representation_size=32, mask_hidden_layers=256, use_mask=False, batch_size=4, box_head_dropout_rate=0, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=2, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_5956_ap_0_3810_test_ar_0_5099_ap_0_3387_epoch25_WithoutClincal_04-29-2022 14-22-30_ov_6]
Best AR validation model has been saved to: [val_ar_0_6824_ap_0_3597_test_ar_0_5849_ap_0_3117_epoch20_WithoutClincal_04-29-2022 14-14-22_ov_6]
The final model has been saved to: [val_ar_0_5324_ap_0_3210_test_ar_0_4868_ap_0_3096_epoch100_WithoutClincal_04-29-2022 16-19-49_ov_6]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990

```

<img width="601" alt="image" src="https://user-images.githubusercontent.com/37566901/165916522-f24c2d39-5517-4cca-b777-43af40c91c33.png">


### dropout=0.2, with ReduceLROnPlateau, factor=0.1, patience=3 (v=0.2702, t=0.3200, best_v=0.3149, best_t=0.2506)

```
========================================For Training [ov_3]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='ov_3', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, using_fpn=False, representation_size=32, mask_hidden_layers=256, use_mask=False, batch_size=4, box_head_dropout_rate=0.2, warmup_epochs=20, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_4274_ap_0_3149_test_ar_0_3708_ap_0_2506_epoch21_WithoutClincal_04-29-2022 16-56-55_ov_3]
Best AR validation model has been saved to: [val_ar_0_4717_ap_0_2915_test_ar_0_4333_ap_0_3212_epoch30_WithoutClincal_04-29-2022 17-11-14_ov_3]
The final model has been saved to: [val_ar_0_4267_ap_0_2702_test_ar_0_4190_ap_0_3200_epoch100_WithoutClincal_04-29-2022 19-00-14_ov_3]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```

<img width="598" alt="image" src="https://user-images.githubusercontent.com/37566901/165917316-48aaa1d9-4791-440b-9962-67d8ba3d852c.png">

### baseline no pretrained, no lr_scheduler, no clinical (v=0.1380, t=0.1039, best_v=0.2378, best_t=0.1999)

```
========================================For Training [ov_14]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='ov_14', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=False, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
====================================================================================================

Best AP validation model has been saved to: [val_ar_0_5791_ap_0_2378_test_ar_0_4083_ap_0_1999_epoch35_WithoutClincal_05-01-2022 21-00-29_ov_14]
Best AR validation model has been saved to: [val_ar_0_5791_ap_0_2378_test_ar_0_4083_ap_0_1999_epoch35_WithoutClincal_05-01-2022 21-00-29_ov_14]
The final model has been saved to: [val_ar_0_2489_ap_0_1380_test_ar_0_2617_ap_0_1039_epoch100_WithoutClincal_05-01-2022 22-39-43_ov_14]

====================================================================================================
Load custom model
Not using pretrained backbone.
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```

<img width="592" alt="image" src="https://user-images.githubusercontent.com/37566901/166155821-92ac1245-0ee4-47e7-bb4b-944bfbbaad71.png">
<img width="615" alt="image" src="https://user-images.githubusercontent.com/37566901/166155824-bdafe990-c8f2-4afa-8728-2f746ff52da2.png">

# Model with clinical

### baseline, clinical (v=0.2751, t=0.3412, best_v=0.3625, best_t=0.2484)
```
========================================For Training [mobilenet_v3_lr_schedule]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='mobilenet_v3_lr_schedule', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
=======================================================================================================================

Best AP validation model has been saved to: [val_ar_0_8819_ap_0_3625_test_ar_0_7579_ap_0_2484_epoch1_WithClincal_04-30-2022 09-33-40_mobilenet_v3_lr_schedule]
Best AR validation model has been saved to: [val_ar_0_8819_ap_0_3625_test_ar_0_7579_ap_0_2484_epoch1_WithClincal_04-30-2022 09-33-40_mobilenet_v3_lr_schedule]
The final model has been saved to: [val_ar_0_6258_ap_0_2751_test_ar_0_6477_ap_0_3412_epoch100_WithClincal_04-30-2022 12-14-12_mobilenet_v3_lr_schedule]

=======================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="607" alt="image" src="https://user-images.githubusercontent.com/37566901/166100120-fa58f2f0-4054-4993-a2d7-e59f6d1e2170.png">


### MultiStepLR, clinical (v=0.4273, t=0.4850, best_v=0.6074, best_t=0.5572)
```
========================================For Training [ov_1]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_1', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='MultiStepLR', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_8788_ap_0_6074_test_ar_0_8079_ap_0_5572_epoch13_WithClincal_04-30-2022 12-38-30_ov_1]
Best AR validation model has been saved to: [val_ar_0_8788_ap_0_6074_test_ar_0_8079_ap_0_5572_epoch13_WithClincal_04-30-2022 12-38-30_ov_1]
The final model has been saved to: [val_ar_0_7270_ap_0_4273_test_ar_0_7333_ap_0_4850_epoch100_WithClincal_04-30-2022 14-59-10_ov_1]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="595" alt="image" src="https://user-images.githubusercontent.com/37566901/166100129-567b7d27-5bb9-4a21-9907-4dafcd3227b2.png">


### larger, clinical (v=0.3586, t=0.3666, best_v=0.4167, best_t=0.3803)

```
========================================For Training [ov_2]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_2', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=32, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_7766_ap_0_4167_test_ar_0_7418_ap_0_3803_epoch2_WithClincal_04-30-2022 15-05-24_ov_2]
Best AR validation model has been saved to: [val_ar_0_7994_ap_0_3201_test_ar_0_7666_ap_0_2632_epoch1_WithClincal_04-30-2022 15-03-37_ov_2]
The final model has been saved to: [val_ar_0_7215_ap_0_3586_test_ar_0_6976_ap_0_3666_epoch100_WithClincal_04-30-2022 17-43-02_ov_2]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,412,279
[model.backbone]: 1,092,928
[model.rpn]: 11,723
[model.roi_heads]: 106,526
[model.roi_heads.box_head]: 104,576
[model.roi_heads.box_head.fc6]: 100,416
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.clinical_convs]: 46,560
[model.fuse_convs]: 46,464
```

<img width="609" alt="image" src="https://user-images.githubusercontent.com/37566901/166100139-37499643-957f-4d36-87b0-7b7866d01e97.png">

### 4 doprout = 0.2, clinincal (v=0.2332, t=0.2851, best_v=0.3249, best_t=0.2680)

```
========================================For Training [ov_3]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_3', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0.2, clinical_conv_dropout_rate=0.2, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0.2, box_head_dropout_rate=0.2, fuse_depth=4)
===================================================================================================

Best AP validation model has been saved to: [val_ar_0_8248_ap_0_3249_test_ar_0_7416_ap_0_2680_epoch1_WithClincal_04-30-2022 19-37-05_ov_3]
Best AR validation model has been saved to: [val_ar_0_8248_ap_0_3249_test_ar_0_7416_ap_0_2680_epoch1_WithClincal_04-30-2022 19-37-05_ov_3]
The final model has been saved to: [val_ar_0_5447_ap_0_2332_test_ar_0_6017_ap_0_2851_epoch100_WithClincal_04-30-2022 22-15-09_ov_3]

===================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="605" alt="image" src="https://user-images.githubusercontent.com/37566901/166105365-6fc4b1ab-c37b-4756-9516-9a7d950ac4b4.png">


### dropout=0 clinical baseline, no lr_scheduler. (v=0.3854, t=0.4183,, best_v=0.5869, best_t=0.5425)

```
========================================For Training [mobilenet_v3_no_pretrained_no_lr_scheduler]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='mobilenet_v3_no_pretrained_no_lr_scheduler', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
=========================================================================================================================================

Best AP validation model has been saved to: [val_ar_0_8617_ap_0_5869_test_ar_0_8083_ap_0_5425_epoch53_WithClincal_05-01-2022 01-30-47_mobilenet_v3_no_pretrained_no_lr_scheduler]
Best AR validation model has been saved to: [val_ar_0_8962_ap_0_3584_test_ar_0_7726_ap_0_2714_epoch1_WithClincal_05-01-2022 00-07-04_mobilenet_v3_no_pretrained_no_lr_scheduler]
The final model has been saved to: [val_ar_0_8038_ap_0_3854_test_ar_0_6872_ap_0_4183_epoch100_WithClincal_05-01-2022 02-46-54_mobilenet_v3_no_pretrained_no_lr_scheduler]

=========================================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="396" alt="image" src="https://user-images.githubusercontent.com/37566901/166115022-5d545d84-7646-4128-b365-dbe4107c8515.png">

### lr=1e-5, clinical (v= 0.2693, t=0.2759, best_v=0.3427, best_t=0.2859)

```
========================================For Training [mobilenet_v3_no_pretrained_no_lr_scheduler]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='mobilenet_v3_no_pretrained_no_lr_scheduler', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=1e-05, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
=========================================================================================================================================

Best AP validation model has been saved to: [val_ar_0_8951_ap_0_3427_test_ar_0_7418_ap_0_2859_epoch50_WithClincal_05-01-2022 05-44-29_mobilenet_v3_no_pretrained_no_lr_scheduler]
Best AR validation model has been saved to: [val_ar_0_8951_ap_0_3296_test_ar_0_7686_ap_0_2882_epoch42_WithClincal_05-01-2022 05-31-36_mobilenet_v3_no_pretrained_no_lr_scheduler]
The final model has been saved to: [val_ar_0_8626_ap_0_2693_test_ar_0_7668_ap_0_2759_epoch70_WithClincal_05-01-2022 06-16-19_mobilenet_v3_no_pretrained_no_lr_scheduler]

=========================================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```

<img width="400" alt="image" src="https://user-images.githubusercontent.com/37566901/166121337-eb823b0d-3b1e-4c1a-a7fd-f2d9a512a670.png">


### clinical, ReduceLROnPlateau, large paitence = 10, factor = 0.1, lr=1e-1 (v=0.3431, t=0.3360, best_v=0.3944, best_t=0.3769)

```
========================================For Training [ov_10]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_10', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.1, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
====================================================================================================

Best AP validation model has been saved to: [val_ar_0_5982_ap_0_3944_test_ar_0_6331_ap_0_3769_epoch22_WithClincal_05-01-2022 09-58-17_ov_10]
Best AR validation model has been saved to: [val_ar_0_7551_ap_0_3374_test_ar_0_7243_ap_0_3830_epoch39_WithClincal_05-01-2022 10-25-21_ov_10]
The final model has been saved to: [val_ar_0_6982_ap_0_3431_test_ar_0_6581_ap_0_3360_epoch100_WithClincal_05-01-2022 12-01-13_ov_10]

====================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="601" alt="image" src="https://user-images.githubusercontent.com/37566901/166155516-54f97abf-8f58-4394-a8e7-ffe2ec2a922e.png">
<img width="622" alt="image" src="https://user-images.githubusercontent.com/37566901/166155519-5c47c2f4-5805-4db5-bd20-ff82fafdd12c.png">

### larger lr = 0.1 (v=0.0544, t=0.0708, best_v=0.4028, best_t=0.3968, best_v=0.4028, best_t=0.3968) 
```
========================================For Training [ov_11]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_11', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.1, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
====================================================================================================

Best AP validation model has been saved to: [val_ar_0_6897_ap_0_4028_test_ar_0_6099_ap_0_3968_epoch68_WithClincal_05-01-2022 13-51-48_ov_11]
Best AR validation model has been saved to: [val_ar_0_6897_ap_0_4028_test_ar_0_6099_ap_0_3968_epoch68_WithClincal_05-01-2022 13-51-48_ov_11]
The final model has been saved to: [val_ar_0_3222_ap_0_0544_test_ar_0_3150_ap_0_0708_epoch100_WithClincal_05-01-2022 14-42-00_ov_11]

====================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/37566901/166155601-c44a2f32-1b03-4278-95bc-c50bc3739c45.png">
<img width="619" alt="image" src="https://user-images.githubusercontent.com/37566901/166155608-aa600d77-8643-4808-a521-24c268a07d66.png">


## samller lr = 1e-3 (v= 0.3869, t=0.4522, best_v=0.4498, best_t=0.4252)
```
========================================For Training [ov_12]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_12', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
====================================================================================================

Best AP validation model has been saved to: [val_ar_0_7641_ap_0_4498_test_ar_0_7120_ap_0_4252_epoch81_WithClincal_05-01-2022 16-52-32_ov_12]
Best AR validation model has been saved to: [val_ar_0_8322_ap_0_3391_test_ar_0_8043_ap_0_3448_epoch7_WithClincal_05-01-2022 14-56-30_ov_12]
The final model has been saved to: [val_ar_0_7397_ap_0_3869_test_ar_0_6872_ap_0_4522_epoch100_WithClincal_05-01-2022 17-22-23_ov_12]

====================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```

<img width="591" alt="image" src="https://user-images.githubusercontent.com/37566901/166155657-7c82c0f4-964a-44af-a4b4-90826ae20ed5.png">
<img width="619" alt="image" src="https://user-images.githubusercontent.com/37566901/166155661-e3b3d952-ae70-4ef8-96df-7ef1f548b827.png">

## samller lr = 1e-3, 200 epoch (v= 0.3869, t=0.4522, best_v=0.4498, best_t=0.4252)

```
========================================For Training [ov_12]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_12', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
====================================================================================================

Best AP validation model has been saved to: [val_ar_0_7222_ap_0_4626_test_ar_0_6727_ap_0_4956_epoch111_WithClincal_05-03-2022 20-39-41_ov_12]
Best AR validation model has been saved to: [val_ar_0_8322_ap_0_3391_test_ar_0_8043_ap_0_3448_epoch7_WithClincal_05-01-2022 14-56-30_ov_12]
The final model has been saved to: [val_ar_0_6650_ap_0_3870_test_ar_0_6604_ap_0_4493_epoch200_WithClincal_05-03-2022 23-04-30_ov_12]

====================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```

<img width="596" alt="image" src="https://user-images.githubusercontent.com/37566901/166484686-fcbed325-720f-47e9-959c-594cf2d94e8b.png">
<img width="622" alt="image" src="https://user-images.githubusercontent.com/37566901/166484793-401e90eb-2b04-4ff2-bd6a-afb62c36a148.png">


## smaller lr = 1e-4 (v=0.3165, t=0.3827, best_v=0.4386, best_t=0.4241)
```
========================================For Training [ov_13]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_13', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0.0001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
====================================================================================================

Best AP validation model has been saved to: [val_ar_0_6977_ap_0_4386_test_ar_0_7352_ap_0_4241_epoch59_WithClincal_05-01-2022 18-58-42_ov_13]
Best AR validation model has been saved to: [val_ar_0_8891_ap_0_2882_test_ar_0_8043_ap_0_3033_epoch1_WithClincal_05-01-2022 17-26-43_ov_13]
The final model has been saved to: [val_ar_0_5733_ap_0_3165_test_ar_0_6015_ap_0_3827_epoch100_WithClincal_05-01-2022 20-02-54_ov_13]

====================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/37566901/166155731-7370ad75-02c7-41e0-a729-9973cfadc370.png">
<img width="624" alt="image" src="https://user-images.githubusercontent.com/37566901/166155736-f4249b0c-bf1b-4904-a130-a5d4cd7d9b7d.png">




### dropout=0.2, no lr_scheduler (v=0.3952, t=0.4149, best_v=0.4468, best_t=0.4720)

```
========================================For Training [ov_15]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_15', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=3, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0.2, clinical_conv_dropout_rate=0.2, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0.2, box_head_dropout_rate=0.2, fuse_depth=4)
====================================================================================================

Best AP validation model has been saved to: [val_ar_0_6951_ap_0_4468_test_ar_0_6640_ap_0_4720_epoch23_WithClincal_05-01-2022 23-17-46_ov_15]
Best AR validation model has been saved to: [val_ar_0_8237_ap_0_2652_test_ar_0_7436_ap_0_2343_epoch1_WithClincal_05-01-2022 22-43-56_ov_15]
The final model has been saved to: [val_ar_0_7161_ap_0_3952_test_ar_0_6208_ap_0_4149_epoch100_WithClincal_05-02-2022 01-13-35_ov_15]

====================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```

<img width="594" alt="image" src="https://user-images.githubusercontent.com/37566901/166156142-5e573765-4674-42fd-9888-22ca295c966f.png">
<img width="611" alt="image" src="https://user-images.githubusercontent.com/37566901/166156148-34a03944-912a-460c-8748-b2526a432888.png">


### clinical, ReduceLROnPlateau, patience=10, factor=0.1, lr=1e-2
```
========================================For Training [ov_10]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='ov_10', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
====================================================================================================

Best AP validation model has been saved to: [val_ar_0_7897_ap_0_5288_test_ar_0_7368_ap_0_5184_epoch42_WithClincal_05-02-2022 07-53-16_ov_10]
Best AR validation model has been saved to: [val_ar_0_8902_ap_0_4305_test_ar_0_7408_ap_0_4547_epoch49_WithClincal_05-02-2022 08-04-27_ov_10]
The final model has been saved to: [val_ar_0_6371_ap_0_3997_test_ar_0_6710_ap_0_4297_epoch100_WithClincal_05-02-2022 09-24-53_ov_10]

====================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="593" alt="image" src="https://user-images.githubusercontent.com/37566901/166170118-a9066178-d0c6-4e50-a168-68398b6655bc.png">
<img width="607" alt="image" src="https://user-images.githubusercontent.com/37566901/166170126-a0781844-b031-4bc0-a8ba-72f802490084.png">

## Fusioon = Add + Residule

```
========================================For Training [with_clinical_add_res]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_add_res', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='add', fusion_residule=True)
====================================================================================================================

Best AP validation model has been saved to: [val_ar_0_6089_ap_0_3359_test_ar_0_4243_ap_0_1894_epoch13_WithClincal_05-04-2022 19-00-14_with_clinical_add_res]
Best AR validation model has been saved to: [val_ar_0_6750_ap_0_2681_test_ar_0_5529_ap_0_2593_epoch26_WithClincal_05-04-2022 19-21-29_with_clinical_add_res]
The final model has been saved to: [val_ar_0_4838_ap_0_2657_test_ar_0_5138_ap_0_2803_epoch100_WithClincal_05-04-2022 21-20-31_with_clinical_add_res]

====================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,218,695
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 27,984
```

<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/166725411-850bed76-cdd9-477e-80e9-c62c12403fb3.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/166725442-a2117604-2c89-45de-a894-3610030cae1b.png">


## Fusion = Add
```
========================================For Training [with_clinical_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='add', fusion_residule=False)
================================================================================================================

Best AP validation model has been saved to: [val_ar_0_8331_ap_0_5238_test_ar_0_7101_ap_0_5071_epoch22_WithClincal_05-04-2022 21-59-26_with_clinical_add]
Best AR validation model has been saved to: [val_ar_0_8799_ap_0_4657_test_ar_0_7976_ap_0_4765_epoch17_WithClincal_05-04-2022 21-51-05_with_clinical_add]
The final model has been saved to: [val_ar_0_6669_ap_0_3626_test_ar_0_6727_ap_0_4698_epoch100_WithClincal_05-05-2022 00-04-48_with_clinical_add]

================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,218,695
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 27,984
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/166725516-f8a1e5e0-842a-40a2-95ea-7dcba80f69fe.png">
<img width="509" alt="image" src="https://user-images.githubusercontent.com/37566901/166725540-45054308-7b04-4ba6-8c02-5d89ea7e5bf2.png">


### Determined comparison.

## With clinical (v=0.4601, t=0.4286, best_v=0.5567, best_t=0.5392)
```
========================================For Training [with_clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
============================================================================================================

Best AP validation model has been saved to: [val_ar_0_7430_ap_0_5567_test_ar_0_7974_ap_0_5392_epoch29_WithClincal_05-02-2022 17-07-17_with_clinical]
Best AR validation model has been saved to: [val_ar_0_8595_ap_0_4564_test_ar_0_7599_ap_0_4436_epoch13_WithClincal_05-02-2022 16-41-36_with_clinical]
The final model has been saved to: [val_ar_0_7269_ap_0_4601_test_ar_0_6710_ap_0_4286_epoch100_WithClincal_05-02-2022 19-03-30_with_clinical]

============================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="595" alt="image" src="https://user-images.githubusercontent.com/37566901/166212138-5199b769-0291-4716-a8ee-6878d98ec6f6.png">
<img width="604" alt="image" src="https://user-images.githubusercontent.com/37566901/166212174-337ff63f-909c-4d01-94c2-1cb4b4fdb79f.png">


## Without clinical (v=0.2754, t=0.2605, best_v=0.2981, best_t=0.2716)
```
========================================For Training [without_clinical]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='without_clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4)
===============================================================================================================

Best AP validation model has been saved to: [val_ar_0_5377_ap_0_2981_test_ar_0_4515_ap_0_2716_epoch30_WithoutClincal_05-02-2022 12-32-56_without_clinical]
Best AR validation model has been saved to: [val_ar_0_6803_ap_0_2134_test_ar_0_5547_ap_0_2393_epoch21_WithoutClincal_05-02-2022 12-19-06_without_clinical]
The final model has been saved to: [val_ar_0_5053_ap_0_2754_test_ar_0_4265_ap_0_2605_epoch100_WithoutClincal_05-02-2022 14-17-08_without_clinical]

===============================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
```
<img width="600" alt="image" src="https://user-images.githubusercontent.com/37566901/166185314-d8daf145-06ec-48e0-8ea6-1a97021b8598.png">
<img width="614" alt="image" src="https://user-images.githubusercontent.com/37566901/166185322-102a9ce8-bd43-471c-b20b-e1d6958e177b.png">

## Depth testing

### fuse_depth=1

```
========================================For Training [with_clinical_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=1, fusion_strategy='add', fusion_residule=False)
================================================================================================================

Best AP validation model has been saved to: [val_ar_0_6286_ap_0_3748_test_ar_0_5761_ap_0_3442_epoch39_WithClincal_05-05-2022 07-42-07_with_clinical_add]
Best AR validation model has been saved to: [val_ar_0_7289_ap_0_3727_test_ar_0_7400_ap_0_4452_epoch45_WithClincal_05-05-2022 07-51-23_with_clinical_add]
The final model has been saved to: [val_ar_0_4670_ap_0_3328_test_ar_0_5388_ap_0_3871_epoch100_WithClincal_05-05-2022 09-14-47_with_clinical_add]

================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,193,063
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/166864512-f55322bf-eea2-4689-9e4b-e2a3db49d164.png">
<img width="513" alt="image" src="https://user-images.githubusercontent.com/37566901/166864527-638d7346-5538-48b6-a389-4d32300f0fdf.png">


### fuse_depth=2
```
========================================For Training [with_clinical_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=2, fusion_strategy='add', fusion_residule=False)
================================================================================================================

Best AP validation model has been saved to: [val_ar_0_7931_ap_0_5010_test_ar_0_6710_ap_0_4713_epoch35_WithClincal_05-05-2022 10-12-54_with_clinical_add]
Best AR validation model has been saved to: [val_ar_0_8059_ap_0_4349_test_ar_0_7208_ap_0_4084_epoch22_WithClincal_05-05-2022 09-52-41_with_clinical_add]
The final model has been saved to: [val_ar_0_5107_ap_0_3767_test_ar_0_4820_ap_0_3465_epoch100_WithClincal_05-05-2022 11-53-20_with_clinical_add]

================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,200,071
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 9,360
```

<img width="494" alt="image" src="https://user-images.githubusercontent.com/37566901/166864610-9f3018f2-ed25-4ed1-bea0-d302400c2168.png">
<img width="506" alt="image" src="https://user-images.githubusercontent.com/37566901/166864622-ce2e3b17-a04d-4afa-a238-11d1499cb679.png">

### fuse_depth=3

```
========================================For Training [with_clinical_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=3, fusion_strategy='add', fusion_residule=False)
================================================================================================================

Best AP validation model has been saved to: [val_ar_0_8930_ap_0_5200_test_ar_0_7726_ap_0_5270_epoch37_WithClincal_05-05-2022 12-55-52_with_clinical_add]
Best AR validation model has been saved to: [val_ar_0_8930_ap_0_5200_test_ar_0_7726_ap_0_5270_epoch37_WithClincal_05-05-2022 12-55-52_with_clinical_add]
The final model has been saved to: [val_ar_0_6718_ap_0_4279_test_ar_0_6604_ap_0_4507_epoch100_WithClincal_05-05-2022 14-36-11_with_clinical_add]

================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,209,383
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 18,672
```
<img width="496" alt="image" src="https://user-images.githubusercontent.com/37566901/166864643-0af2dd66-0ae3-49e4-b7ae-b97a698d724a.png">
<img width="508" alt="image" src="https://user-images.githubusercontent.com/37566901/166864659-e51e88ad-3993-484f-8ee9-17810f1a53e6.png">

## fuse_depth = 10

```
========================================For Training [with_clinical_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=10, fusion_strategy='add', fusion_residule=False)
================================================================================================================

Best AP validation model has been saved to: [val_ar_0_8083_ap_0_4401_test_ar_0_7936_ap_0_4314_epoch2_WithClincal_05-05-2022 17-09-57_with_clinical_add]
Best AR validation model has been saved to: [val_ar_0_8083_ap_0_4401_test_ar_0_7936_ap_0_4314_epoch2_WithClincal_05-05-2022 17-09-56_with_clinical_add]
The final model has been saved to: [val_ar_0_6419_ap_0_3024_test_ar_0_5997_ap_0_3120_epoch100_WithClincal_05-05-2022 19-48-56_with_clinical_add]

================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,274,567
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 83,856
```

<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/166902573-3038105e-afa3-4c72-a4ed-c95ab0452859.png">
<img width="504" alt="image" src="https://user-images.githubusercontent.com/37566901/166902598-58849084-bdac-438a-a438-68c06cf542d2.png">

### fuse_dpeth = 10

```
========================================For Training [with_clinical_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=10, fusion_strategy='add', fusion_residule=False)
================================================================================================================

Best AP validation model has been saved to: [val_ar_0_7590_ap_0_5117_test_ar_0_6563_ap_0_4799_epoch60_WithClincal_05-05-2022 21-52-09_with_clinical_add]
Best AR validation model has been saved to: [val_ar_0_8491_ap_0_3379_test_ar_0_7045_ap_0_2990_epoch10_WithClincal_05-05-2022 20-25-34_with_clinical_add]
The final model has been saved to: [val_ar_0_6297_ap_0_3327_test_ar_0_5460_ap_0_3843_epoch100_WithClincal_05-05-2022 23-01-07_with_clinical_add]

================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,274,567
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 83,856
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/166982459-2b67eeed-d773-4b35-86f3-ffa085ef1b29.png">
<img width="513" alt="image" src="https://user-images.githubusercontent.com/37566901/166982478-e1c88d67-6526-4960-8d23-c6252ced0883.png">

### fuse_depth = 20

```
========================================For Training [with_clinical_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.01, weight_decay=0.001, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=20, fusion_strategy='add', fusion_residule=False)
================================================================================================================

Best AP validation model has been saved to: [val_ar_0_8326_ap_0_5104_test_ar_0_7495_ap_0_4951_epoch13_WithClincal_05-05-2022 23-27-36_with_clinical_add]
Best AR validation model has been saved to: [val_ar_0_8326_ap_0_5104_test_ar_0_7495_ap_0_4951_epoch13_WithClincal_05-05-2022 23-27-36_with_clinical_add]
The final model has been saved to: [val_ar_0_7654_ap_0_4646_test_ar_0_6226_ap_0_4319_epoch100_WithClincal_05-06-2022 01-58-50_with_clinical_add]

================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,367,687
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 176,976
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/166982545-10947638-c95c-4279-82a7-8f0177767b3c.png">
<img width="512" alt="image" src="https://user-images.githubusercontent.com/37566901/166982566-757c2243-9999-44aa-8e5a-998400f96c28.png">

### fuse_depth = 20
fail to run.



## Improvement:
```
========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_9667_ap_0_6658_test_ar_0_8724_ap_0_5694_epoch13_WithClincal_05-13-2022 22-18-31_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6344_test_ar_0_8704_ap_0_5807_epoch12_WithClincal_05-13-2022 22-16-49_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_8055_ap_0_3879_test_ar_0_6888_ap_0_4654_epoch32_WithClincal_05-13-2022 22-47-35_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/168286863-37cfe110-4442-4f1c-a987-f92e316e55ad.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/168286905-dbcc85f5-b584-47c6-a672-2520fba6a65b.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/168286929-2dd83097-e2a0-4e41-a176-336a7863b7c6.png">

### Remove loss_objectness and loss_rpn_box_reg

```
========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_9667_ap_0_6636_test_ar_0_8724_ap_0_6179_epoch22_WithClincal_05-13-2022 23-32-16_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6511_test_ar_0_8474_ap_0_5885_epoch21_WithClincal_05-13-2022 23-30-25_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_9667_ap_0_6359_test_ar_0_8724_ap_0_6534_epoch29_WithClincal_05-13-2022 23-43-08_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```

<img width="495" alt="image" src="https://user-images.githubusercontent.com/37566901/168297303-b7396dd4-ae88-46c7-b3c8-db2949377a4c.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/168297344-bfc20b0e-1202-4f3e-bd09-d17ee494e257.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/168297391-bce1e8f1-91fe-42d7-805a-eabeb1be7f4d.png">

### with lr_scheduler

```
========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_9556_ap_0_7105_test_ar_0_8599_ap_0_5995_epoch39_WithClincal_05-14-2022 00-54-45_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6357_test_ar_0_8704_ap_0_5953_epoch12_WithClincal_05-14-2022 00-10-14_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_9556_ap_0_6799_test_ar_0_8331_ap_0_6038_epoch49_WithClincal_05-14-2022 01-10-22_with_clinical_residule_add]

=========================================================================================================================
Load custom model
```
<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/168313926-fa8fedfc-167f-4d85-a4e4-a2fb6aa56e2a.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/168314012-15b223b1-2d91-4eae-ae23-cbad7d09a19b.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/168314035-99124547-446e-48dc-8722-19534c5035e3.png">

### lr = 1e-3, no lr_scheduler,  (has overfitting.)
```
weight:

loss_dict["loss_classifier"] *= 10
loss_dict["loss_box_reg"] *= 5

loss_dict["loss_objectness"] *= 0.01
loss_dict["loss_rpn_box_reg"] *= 0.01

========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_8830_ap_0_7316_test_ar_0_6420_ap_0_5311_epoch184_WithClincal_05-14-2022 06-12-26_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6237_test_ar_0_8599_ap_0_6080_epoch15_WithClincal_05-14-2022 01-41-04_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_7686_ap_0_5797_test_ar_0_7188_ap_0_5458_epoch261_WithClincal_05-14-2022 08-15-49_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/37566901/168397278-a39ccd69-a28b-4fed-8dcf-4f7d0959535b.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/168397286-1af7d4e7-a2aa-4719-836b-fdb8bf853a8e.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/168397295-b9bda92b-e219-49e9-be69-9f382470acef.png">

### reduceLROnPlateau_factor=0.5

```
    loss_dict["loss_classifier"] *= 10
    loss_dict["loss_box_reg"] *= 5

    loss_dict["loss_objectness"] *= 1e-5
    loss_dict["loss_rpn_box_reg"] *= 1e-5
    
========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.5, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_9513_ap_0_6851_test_ar_0_8099_ap_0_6636_epoch36_WithClincal_05-14-2022 10-07-35_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6114_test_ar_0_8829_ap_0_5572_epoch10_WithClincal_05-14-2022 09-26-16_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_9026_ap_0_5824_test_ar_0_8331_ap_0_6114_epoch168_WithClincal_05-14-2022 13-26-02_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="496" alt="image" src="https://user-images.githubusercontent.com/37566901/168409144-e7661808-3555-4fa7-a0b3-1536ae6c0ecd.png">
<img width="515" alt="image" src="https://user-images.githubusercontent.com/37566901/168409141-b81d483a-c66d-4d6d-90e1-bed06483c6a9.png">
<img width="509" alt="image" src="https://user-images.githubusercontent.com/37566901/168409149-0da06ebc-616e-4228-bc04-9bc98132cf1f.png">

### no lr_scheduler

```

========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.5, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_9667_ap_0_6943_test_ar_0_8224_ap_0_6060_epoch12_WithClincal_05-14-2022 13-53-26_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6581_test_ar_0_8724_ap_0_5616_epoch10_WithClincal_05-14-2022 13-50-07_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_8077_ap_0_6358_test_ar_0_7743_ap_0_6053_epoch100_WithClincal_05-14-2022 17-13-50_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="496" alt="image" src="https://user-images.githubusercontent.com/37566901/168415346-a13b0e4a-6f29-4829-98de-0780d541c77e.png">
<img width="521" alt="image" src="https://user-images.githubusercontent.com/37566901/168415355-3f67abe2-f3c3-4259-905f-97639e8428ce.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/168415360-f9e59c81-5173-4334-bbe4-b820c140ba18.png">

### no lr_scheduler, no loss_objectnees and loss_rpn_box_reg

```
def loss_multiplier(loss_dict):

    loss_dict["loss_classifier"] *= 10
    loss_dict["loss_box_reg"] *= 5

    # loss_dict["loss_objectness"] *= 1e-5
    # loss_dict["loss_rpn_box_reg"] *= 1e-5

    loss_dict["loss_objectness"] = loss_dict["loss_objectness"].detach()
    loss_dict["loss_rpn_box_reg"]  = loss_dict["loss_rpn_box_reg"].detach()

    return loss_dict
    
    
========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_9444_ap_0_6703_test_ar_0_8313_ap_0_5836_epoch53_WithClincal_05-14-2022 20-22-10_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6249_test_ar_0_8474_ap_0_5713_epoch11_WithClincal_05-14-2022 19-16-09_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_8905_ap_0_5704_test_ar_0_7688_ap_0_6188_epoch100_WithClincal_05-14-2022 21-36-43_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="495" alt="image" src="https://user-images.githubusercontent.com/37566901/168424171-53fc42fc-c045-48c1-b3db-c439ddabd68a.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/168424173-4c8c3fdb-c1de-4dfd-a569-523a56cd84f8.png">
<img width="523" alt="image" src="https://user-images.githubusercontent.com/37566901/168424182-84d42d41-244b-47f4-a679-4b649c165505.png">

### no lr_scheduler (with_clinical, test_ap = 0.6181)

```
    loss_dict["loss_classifier"] *= 10
    loss_dict["loss_box_reg"] *= 5

    loss_dict["loss_objectness"] *= 1e-5
    loss_dict["loss_rpn_box_reg"] *= 1e-5


========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_9667_ap_0_7154_test_ar_0_8099_ap_0_6181_epoch32_WithClincal_05-15-2022 09-02-41_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6436_test_ar_0_8599_ap_0_5858_epoch14_WithClincal_05-15-2022 08-33-18_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_9413_ap_0_6401_test_ar_0_8599_ap_0_6040_epoch50_WithClincal_05-15-2022 09-30-31_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```

<img width="495" alt="image" src="https://user-images.githubusercontent.com/37566901/168453577-815cdfc4-6751-4198-b7f5-bcd5dcb0a696.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/168453581-16a9b9cc-98d5-46a7-aa33-6f3c8f408a72.png">
<img width="523" alt="image" src="https://user-images.githubusercontent.com/37566901/168453587-18e6537d-7a76-4709-ab3d-33f18cf7a623.png">


### no lr_scheduler (without clinical, test_ap=0.3685)
```
    loss_dict["loss_classifier"] *= 10
    loss_dict["loss_box_reg"] *= 5

    loss_dict["loss_objectness"] *= 1e-5
    loss_dict["loss_rpn_box_reg"] *= 1e-5
   
========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_6396_ap_0_4318_test_ar_0_6117_ap_0_3685_epoch27_WithoutClincal_05-15-2022 10-15-38_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_6664_ap_0_3854_test_ar_0_5726_ap_0_4147_epoch44_WithoutClincal_05-15-2022 10-41-27_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_4986_ap_0_3221_test_ar_0_3999_ap_0_2915_epoch50_WithoutClincal_05-15-2022 10-50-43_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
    
```

<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/168453602-eeffba09-7449-49de-9dcd-d0a484ea6dda.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/168453616-db1e24f2-8afa-474d-b69c-84a3cb60829e.png">
<img width="515" alt="image" src="https://user-images.githubusercontent.com/37566901/168453619-520d3d98-5e8c-43b5-8b3d-c047ccf3aa03.png">


## losses weighting results 

### CXR+Clinical (test_ap=6808)

```
loss_dict["loss_classifier"] *= 10
loss_dict["loss_box_reg"] *= 5

loss_dict["loss_objectness"] *= 1e-5
loss_dict["loss_rpn_box_reg"] *= 1e-5
========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_9381_ap_0_6799_test_ar_0_8724_ap_0_6287_epoch33_WithClincal_05-15-2022 12-41-36_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6284_test_ar_0_8579_ap_0_5823_epoch10_WithClincal_05-15-2022 12-04-40_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_9667_ap_0_6751_test_ar_0_8724_ap_0_6806_epoch50_WithClincal_05-15-2022 13-10-34_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="495" alt="image" src="https://user-images.githubusercontent.com/37566901/168459788-924ac0a2-d2dd-46aa-b6f7-7df8e99559d5.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/168459805-2406eced-4ea1-4ef1-9b9e-fc9d748314d8.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/168459812-2a065da3-4d03-4c0f-901c-daf802b2f72d.png">

### CXR (4076)
```
loss_dict["loss_classifier"] *= 10
loss_dict["loss_box_reg"] *= 5

loss_dict["loss_objectness"] *= 1e-5
loss_dict["loss_rpn_box_reg"] *= 1e-5
========================================For Training [with_clinical_residule_add]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='with_clinical_residule_add', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
=========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_5877_ap_0_4511_test_ar_0_5102_ap_0_4046_epoch44_WithoutClincal_05-15-2022 14-28-44_with_clinical_residule_add]
Best AR validation model has been saved to: [val_ar_0_6353_ap_0_3862_test_ar_0_4727_ap_0_3300_epoch18_WithoutClincal_05-15-2022 13-44-41_with_clinical_residule_add]
The final model has been saved to: [val_ar_0_5552_ap_0_4397_test_ar_0_4852_ap_0_4076_epoch50_WithoutClincal_05-15-2022 14-38-33_with_clinical_residule_add]

=========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```
<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/168459831-0d2548b8-1231-473d-b7ec-b5fc3f19907d.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/168459834-8b8f4fea-1927-495f-9967-2fac68ab64ae.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/168459841-9ac965ba-baa2-4c43-a044-b781f746f646.png">

## work 2:
## CXR+Clinical(0.6275)
```
========================================For Training [CXR+Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR+Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=20, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_9270_ap_0_6763_test_ar_0_8599_ap_0_6275_epoch27_WithClincal_05-15-2022 18-10-01_CXR+Clinical]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_5999_test_ar_0_8724_ap_0_5648_epoch10_WithClincal_05-15-2022 17-41-35_CXR+Clinical]
The final model has been saved to: [val_ar_0_9413_ap_0_6259_test_ar_0_8724_ap_0_6139_epoch50_WithClincal_05-15-2022 18-48-12_CXR+Clinical]

===========================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592

    loss_dict["loss_classifier"] *= 10
    loss_dict["loss_box_reg"] *= 5

    loss_dict["loss_objectness"] *= 1e-5
    loss_dict["loss_rpn_box_reg"] *= 1e-5

```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/168468096-176a7529-bd75-4c9a-818a-f2e96be41c36.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/168468104-490e85ff-31a8-4a65-b430-aa1efcd580b5.png">
<img width="515" alt="image" src="https://user-images.githubusercontent.com/37566901/168468115-ec3f5d79-5d48-484d-8978-1aa3c985bf84.png">


### CXR (0.3722)

```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=20, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_5214_ap_0_4189_test_ar_0_4888_ap_0_3722_epoch38_WithoutClincal_05-15-2022 19-54-04_CXR]
Best AR validation model has been saved to: [val_ar_0_6161_ap_0_3791_test_ar_0_5868_ap_0_3426_epoch26_WithoutClincal_05-15-2022 19-33-54_CXR]
The final model has been saved to: [val_ar_0_4196_ap_0_3669_test_ar_0_4335_ap_0_3343_epoch50_WithoutClincal_05-15-2022 20-13-24_CXR]

==================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990

    loss_dict["loss_classifier"] *= 10
    loss_dict["loss_box_reg"] *= 5

    loss_dict["loss_objectness"] *= 1e-5
    loss_dict["loss_rpn_box_reg"] *= 1e-5
```
<img width="495" alt="image" src="https://user-images.githubusercontent.com/37566901/168468151-e4d02726-0182-4f8b-946b-8bb52080d52a.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/168468158-3f021df2-4f35-4980-9391-b878e7239c87.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/168468167-75ae154c-7820-40fb-94dc-4c10a07ab991.png">

## Work 3

### CXR + Clinical (6373)

```
========================================For Training [CXR+Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR+Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=20, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_9270_ap_0_7011_test_ar_0_8349_ap_0_6114_epoch35_WithClincal_05-15-2022 22-46-44_CXR+Clinical]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6706_test_ar_0_8724_ap_0_6255_epoch12_WithClincal_05-15-2022 22-08-46_CXR+Clinical]
The final model has been saved to: [val_ar_0_9413_ap_0_6099_test_ar_0_8456_ap_0_6373_epoch50_WithClincal_05-15-2022 23-10-14_CXR+Clinical]

===========================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="499" alt="image" src="https://user-images.githubusercontent.com/37566901/168478403-8d04b2c0-fb2d-41c7-9b39-1f2177093be7.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/168478412-1b073554-acf2-4995-a6b8-e5657be6fc2b.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/168478417-180c811a-4c1a-4de0-82af-b73591e7a2ab.png">

### CXR (0.4302)

```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=20, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_6246_ap_0_4713_test_ar_0_5670_ap_0_4302_epoch41_WithoutClincal_05-16-2022 00-19-11_CXR]
Best AR validation model has been saved to: [val_ar_0_6703_ap_0_4387_test_ar_0_5190_ap_0_3769_epoch36_WithoutClincal_05-16-2022 00-11-05_CXR]
The final model has been saved to: [val_ar_0_5056_ap_0_3320_test_ar_0_4067_ap_0_3387_epoch50_WithoutClincal_05-16-2022 00-33-45_CXR]

==================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/37566901/168478487-a71295b9-ccaf-4009-9542-4ce1a39edca4.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/168478490-a999db38-4bf6-4ebd-a3d4-be29ee8bedb8.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/168478493-badd22a4-954f-4d92-902a-71fb4b053a04.png">



# 10 patience

## Loss weighting
```
loss_dict["loss_classifier"] *= 10
loss_dict["loss_box_reg"] *= 5

loss_dict["loss_objectness"] *= 1e-5
loss_dict["loss_rpn_box_reg"] *= 1e-5
```
## CXR (3713, 2989)
```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_7549_ap_0_5195_test_ar_0_6029_ap_0_3713_epoch22_WithoutClincal_05-16-2022 02-47-55_CXR]
Best AR validation model has been saved to: [val_ar_0_7549_ap_0_5195_test_ar_0_6029_ap_0_3713_epoch22_WithoutClincal_05-16-2022 02-47-55_CXR]
The final model has been saved to: [val_ar_0_5288_ap_0_3648_test_ar_0_4180_ap_0_2989_epoch25_WithoutClincal_05-16-2022 02-52-47_CXR]

==================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```

<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/168498069-f9a7364e-0a3a-49d5-8420-4613615b8e0c.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/168498073-80f974f9-26a0-48af-ae0d-dedb9e45e241.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/168498564-ffacba24-34f4-4ba6-8666-04f7011c274f.png">

## CXR+Clinical(6069, 5755)
```
========================================For Training [CXR+Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR+Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_9667_ap_0_7023_test_ar_0_8474_ap_0_6069_epoch18_WithClincal_05-16-2022 03-27-04_CXR+Clinical]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6642_test_ar_0_8474_ap_0_5746_epoch14_WithClincal_05-16-2022 03-20-26_CXR+Clinical]
The final model has been saved to: [val_ar_0_9127_ap_0_6019_test_ar_0_8349_ap_0_5755_epoch61_WithClincal_05-16-2022 04-37-05_CXR+Clinical]

===========================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```

<img width="496" alt="image" src="https://user-images.githubusercontent.com/37566901/168498747-7481a0c0-975c-4d37-8716-036d6a7bfb2b.png">
<img width="521" alt="image" src="https://user-images.githubusercontent.com/37566901/168498755-4ce3e066-9f66-45ed-b478-b301512698a5.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/168498759-c3def4eb-b9ec-4535-8169-ea19fc2d17fe.png">

# 20 patience
## CXR (3694, 4183)
```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=20, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_6479_ap_0_4566_test_ar_0_5368_ap_0_3694_epoch27_WithoutClincal_05-16-2022 05-25-05_CXR]
Best AR validation model has been saved to: [val_ar_0_7326_ap_0_4072_test_ar_0_5849_ap_0_4103_epoch35_WithoutClincal_05-16-2022 05-38-02_CXR]
The final model has been saved to: [val_ar_0_6277_ap_0_4201_test_ar_0_5583_ap_0_4183_epoch38_WithoutClincal_05-16-2022 05-42-55_CXR]

==================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/168498814-d005caad-a2df-4d73-bee1-1dbd58e545ad.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/168498804-2b05c01e-f976-4591-8b12-342510182e62.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/168498810-216887c8-7973-44e9-b9c3-0a4c76dbf19f.png">

## CXR+Clinical(0.6305, 0.5471)
```
========================================For Training [CXR+Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR+Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=20, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_9556_ap_0_6876_test_ar_0_8438_ap_0_6305_epoch43_WithClincal_05-16-2022 06-58-30_CXR+Clinical]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6373_test_ar_0_8349_ap_0_6264_epoch19_WithClincal_05-16-2022 06-18-25_CXR+Clinical]
The final model has been saved to: [val_ar_0_8259_ap_0_5308_test_ar_0_7456_ap_0_5471_epoch64_WithClincal_05-16-2022 07-33-22_CXR+Clinical]

===========================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="495" alt="image" src="https://user-images.githubusercontent.com/37566901/168498884-e6814625-8dd3-4559-9d40-c972c9a65b97.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/168498888-6540b25c-66b9-4bdb-90cd-805463d4d367.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/168498891-feb523b2-aa78-4210-bc3b-bd61db862c07.png">

## patience = 5

### CXR
```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=5, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_7110_ap_0_4498_test_ar_0_5136_ap_0_3587_epoch17_WithoutClincal_05-16-2022 17-13-28_CXR]
Best AR validation model has been saved to: [val_ar_0_7110_ap_0_4498_test_ar_0_5136_ap_0_3587_epoch17_WithoutClincal_05-16-2022 17-13-28_CXR]
The final model has been saved to: [val_ar_0_5795_ap_0_4115_test_ar_0_4477_ap_0_3555_epoch24_WithoutClincal_05-16-2022 17-24-20_CXR]

==================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```
<img width="330" alt="image" src="https://user-images.githubusercontent.com/37566901/168542037-e2f66b21-c13a-444f-97e4-ce24375e1419.png">
<img width="346" alt="image" src="https://user-images.githubusercontent.com/37566901/168542059-6b34e670-f197-4c6d-bc4a-79bda56bfdf6.png">
<img width="345" alt="image" src="https://user-images.githubusercontent.com/37566901/168542084-c874c723-80f2-4963-a7d6-76cc0fca87f9.png">

### CXR+Clinical

```
========================================For Training [CXR+Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR+Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=5, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_9381_ap_0_6852_test_ar_0_8579_ap_0_5841_epoch8_WithClincal_05-16-2022 16-17-32_CXR+Clinical]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_5986_test_ar_0_8724_ap_0_5948_epoch10_WithClincal_05-16-2022 16-20-56_CXR+Clinical]
The final model has been saved to: [val_ar_0_9413_ap_0_6168_test_ar_0_8349_ap_0_6019_epoch24_WithClincal_05-16-2022 16-43-07_CXR+Clinical]

===========================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="332" alt="image" src="https://user-images.githubusercontent.com/37566901/168542178-748b0e6f-81a4-4630-a589-6012c94bd642.png">
<img width="345" alt="image" src="https://user-images.githubusercontent.com/37566901/168542208-c59e0e43-814c-4ef9-a30a-fc1b9e2a4eb4.png">
<img width="346" alt="image" src="https://user-images.githubusercontent.com/37566901/168542231-8e340310-d435-4327-b0b0-d230b07dad3b.png">



# patience = 30

## CXR + Clinical (0.6776, 0.6534)
```
========================================For Training [CXR+Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR+Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=30, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_9381_ap_0_6945_test_ar_0_8224_ap_0_6378_epoch20_WithClincal_05-16-2022 18-13-10_CXR+Clinical]
Best AR validation model has been saved to: [val_ar_0_9667_ap_0_6652_test_ar_0_8349_ap_0_6278_epoch26_WithClincal_05-16-2022 18-23-31_CXR+Clinical]
The final model has been saved to: [val_ar_0_9556_ap_0_6776_test_ar_0_7849_ap_0_6534_epoch71_WithClincal_05-16-2022 19-38-03_CXR+Clinical]

===========================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,223,303
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 41,904
[model.fuse_convs]: 32,592
```
<img width="495" alt="image" src="https://user-images.githubusercontent.com/37566901/168588982-d5261904-0493-47cc-9f6f-2a4814ac6550.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/168589010-b2cb3a70-68e1-4f7a-b687-3b8b545dfadc.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/168589031-7be013ca-9fe0-415a-a4ce-7a2a142f921b.png">


## CXR (0.4046, 0.3893)

```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=30, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[30, 50, 70, 90], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=32, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='concat', fusion_residule=False)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_7518_ap_0_5042_test_ar_0_5388_ap_0_3255_epoch22_WithoutClincal_05-16-2022 20-18-25_CXR]
Best AR validation model has been saved to: [val_ar_0_7518_ap_0_5042_test_ar_0_5388_ap_0_3255_epoch22_WithoutClincal_05-16-2022 20-18-25_CXR]
The final model has been saved to: [val_ar_0_5966_ap_0_4046_test_ar_0_5388_ap_0_3893_epoch54_WithoutClincal_05-16-2022 21-10-18_CXR]

==================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,040,729
[model.backbone]: 1,009,968
[model.rpn]: 3,595
[model.roi_heads]: 27,166
[model.roi_heads.box_head]: 26,176
[model.roi_heads.box_head.fc6]: 25,120
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```
<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/168589122-9eddf99f-28db-4ce9-bb4d-f3a31c5b9b48.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/168589693-b3223b4f-e5f2-4c03-b3d8-721becbf20f3.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/168589658-2a70ebba-753e-4261-8392-c361f4bee89a.png">

# Correct version:

## Experiment 1
### Setup
```
ModelSetup(
        name="CXR+Clinical",
        use_clinical=True,
        use_custom_model=True,
        use_early_stop_model=True,
        best_ar_val_model_path=None,
        best_ap_val_model_path=None,
        final_model_path=None,
        backbone="mobilenet_v3",
        optimiser="sgd",
        lr=1e-3,
        # lr=1e-4,
        # weight_decay=0.001,
        weight_decay=0,
        pretrained=True,
        record_training_performance=True,
        dataset_mode="unified",
        image_size=256,
        backbone_out_channels=16,
        batch_size=4,
        warmup_epochs=0,
        lr_scheduler="ReduceLROnPlateau",
        # lr_scheduler=None,
        reduceLROnPlateau_factor=0.1,
        reduceLROnPlateau_patience=10,
        reduceLROnPlateau_full_stop=False,
        multiStepLR_milestones=[30, 50, 70, 90],
        multiStepLR_gamma=0.1,
        representation_size=32,
        mask_hidden_layers=256,
        using_fpn=False,
        use_mask=False,
        clinical_expand_dropout_rate=0,
        clinical_conv_dropout_rate=0,
        clinical_input_channels=32,
        clinical_num_len=9,
        clinical_conv_channels=32,
        fuse_conv_channels=32,
        fuse_dropout_rate=0,
        box_head_dropout_rate=0,
        fuse_depth=4,
        fusion_strategy="add",
        fusion_residule=False,
    )

# No loss_multiplier,
# Not using gt in training detector.
# No pretraining epochs for RPN.

```
![image](https://user-images.githubusercontent.com/37566901/168739336-073f8c69-55dd-48b7-a5db-f4304e17d528.png)
![image](https://user-images.githubusercontent.com/37566901/168739385-63085a65-ff3d-4639-8520-27a405c91578.png)
![image](https://user-images.githubusercontent.com/37566901/168739407-07d4c8d4-9744-41d7-aac6-97cad9a1989d.png)
![image](https://user-images.githubusercontent.com/37566901/168739426-11438ef4-b118-4790-83a2-9f3d5e1e1925.png)


## ResNet50, with clinical.

```
========================================For Training [CXR_Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='MultiStepLR', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=500, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=32, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='add', fusion_residule=False)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_5541_ap_0_3371_test_ar_0_4811_ap_0_1730_epoch21_WithClincal_05-18-2022 09-43-35_CXR_Clinical]
Best AR validation model has been saved to: [val_ar_0_5998_ap_0_2663_test_ar_0_4458_ap_0_2243_epoch56_WithClincal_05-18-2022 11-36-38_CXR_Clinical]
The final model has been saved to: [val_ar_0_2478_ap_0_1711_test_ar_0_2610_ap_0_2081_epoch119_WithClincal_05-18-2022 14-58-14_CXR_Clinical]

===========================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Not using pretrained MaksRCNN model.
[model]: 42,150,939
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 808,990
[model.roi_heads.box_head]: 807,040
[model.roi_heads.box_head.fc6]: 802,880
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.clinical_convs]: 3,543,552
[model.fuse_convs]: 2,952,960
```
<img width="408" alt="image" src="https://user-images.githubusercontent.com/37566901/168960866-0b438899-404b-4d54-974b-ae199d997ec5.png">
<img width="510" alt="image" src="https://user-images.githubusercontent.com/37566901/168960898-ba1e1756-dd29-4bcf-b150-d79ecaf1040c.png">
<img width="511" alt="image" src="https://user-images.githubusercontent.com/37566901/168960909-b1c16979-35e1-42a2-9e8e-d768a730549b.png">



## First Exp for real model:

### CXR (0.2274)

```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=64, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='add', fusion_residule=False)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_4734_ap_0_2989_test_ar_0_3947_ap_0_2774_epoch142_WithoutClincal_05-19-2022 00-56-01_CXR]
Best AR validation model has been saved to: [val_ar_0_5440_ap_0_2776_test_ar_0_4180_ap_0_2652_epoch55_WithoutClincal_05-18-2022 22-31-47_CXR]
The final model has been saved to: [val_ar_0_2889_ap_0_1583_test_ar_0_2787_ap_0_1881_epoch200_WithoutClincal_05-19-2022 02-31-58_CXR]

==================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Not using pretrained MaksRCNN model.
[model]: 27,796,717
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 403,486
[model.roi_heads.box_head]: 402,496
[model.roi_heads.box_head.fc6]: 401,440
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/169186733-bae1a53d-afa7-4361-b91c-28cda10a0b33.png">
<img width="515" alt="image" src="https://user-images.githubusercontent.com/37566901/169186751-e2035e13-d696-406f-8c31-78c7a3feebbf.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/169186770-672ccdc3-83ff-4662-a23f-9a794f9fad11.png">


### CXR+Clinical (0.2114)
```
========================================For Training [CXR_Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=64, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=4, fusion_strategy='add', fusion_residule=False)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_3410_ap_0_2745_test_ar_0_3106_ap_0_1807_epoch121_WithClincal_05-19-2022 06-41-49_CXR_Clinical]
Best AR validation model has been saved to: [val_ar_0_5888_ap_0_2725_test_ar_0_4495_ap_0_2114_epoch65_WithClincal_05-19-2022 04-49-22_CXR_Clinical]
The final model has been saved to: [val_ar_0_2727_ap_0_1822_test_ar_0_2711_ap_0_1909_epoch200_WithClincal_05-19-2022 09-18-45_CXR_Clinical]

===========================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Not using pretrained MaksRCNN model.
[model]: 40,924,763
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 403,486
[model.roi_heads.box_head]: 402,496
[model.roi_heads.box_head.fc6]: 401,440
[model.roi_heads.box_head.fc7]: 1,056
[model.roi_heads.box_predictor]: 990
[model.clinical_convs]: 3,543,552
[model.fuse_convs]: 2,952,960
```
<img width="503" alt="image" src="https://user-images.githubusercontent.com/37566901/169186894-91abdbef-e202-475a-b308-b8f255badf97.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/169186912-2188eb1f-513c-433f-ae46-56f9d5e8ced3.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/169186925-9de4b5b8-01a4-46af-ad54-452576c7c55b.png">



<!-- =======================================For Training [CXR_Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=256, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=32, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=64, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_3787_ap_0_2241_test_ar_0_2963_ap_0_1684_epoch41_WithClincal_05-19-2022 13-52-41_CXR_Clinical]
Best AR validation model has been saved to: [val_ar_0_5024_ap_0_2223_test_ar_0_4142_ap_0_2391_epoch46_WithClincal_05-19-2022 14-02-15_CXR_Clinical]
The final model has been saved to: [val_ar_0_5024_ap_0_2223_test_ar_0_4142_ap_0_2391_epoch46_WithClincal_05-19-2022 14-02-15_CXR_Clinical]
 -->


### With clinical

```
========================================For Training [CXR_Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=128, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=64, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, fusion_strategy='add', fusion_residule=False, gt_in_train_till=50)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_5219_ap_0_2930_test_ar_0_4219_ap_0_2192_epoch81_WithClincal_05-19-2022 22-13-26_CXR_Clinical]
Best AR validation model has been saved to: [val_ar_0_6249_ap_0_2303_test_ar_0_5745_ap_0_2537_epoch37_WithClincal_05-19-2022 20-27-54_CXR_Clinical]
The final model has been saved to: [val_ar_0_3020_ap_0_1455_test_ar_0_2852_ap_0_1426_epoch100_WithClincal_05-19-2022 22-58-58_CXR_Clinical]

===========================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 62,453,374
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 1,626,142
[model.roi_heads.box_head]: 1,622,272
[model.roi_heads.box_head.fc6]: 1,605,760
[model.roi_heads.box_head.fc7]: 16,512
[model.roi_heads.box_predictor]: 3,870
[model.clinical_convs]: 26,799,296
```
<img width="496" alt="image" src="https://user-images.githubusercontent.com/37566901/169377103-a957ff58-5854-4648-8bbc-cb572220153a.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/169377128-cdd8372e-6228-42e3-a2ad-e4521a2c2658.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/169377158-a12774ac-81ab-4d95-92a0-d488763bbc50.png">

# CXR+Clinical model testing
## Only using the clinical in box pred. (0.2945, 0.2058)
```
========================================For Training [CXR_Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=256, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=30)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_5389_ap_0_2945_test_ar_0_4126_ap_0_2058_epoch24_WithClincal_05-20-2022 16-54-44_CXR_Clinical]
Best AR validation model has been saved to: [val_ar_0_5667_ap_0_2700_test_ar_0_4710_ap_0_2000_epoch38_WithClincal_05-20-2022 17-19-09_CXR_Clinical]
The final model has been saved to: [val_ar_0_3511_ap_0_2119_test_ar_0_3119_ap_0_2048_epoch100_WithClincal_05-20-2022 19-05-07_CXR_Clinical]

===========================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 57,543,579
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,350,558
[model.roi_heads.box_head]: 3,342,848
[model.roi_heads.box_head.fc6]: 3,277,056
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 26,799,296
```
<img width="503" alt="image" src="https://user-images.githubusercontent.com/37566901/169541438-4aaa866a-7d4e-487c-9322-15a13f827ada.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/169541468-bd24362c-f144-45da-b828-60bcc349e8ed.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/169541487-2f984f7c-6f42-40ec-a83f-5ee097c14191.png">

## using both box pred and spatialisation. (0.3427, 0.2448)
```
========================================For Training [CXR_Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=256, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=30)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_5558_ap_0_3427_test_ar_0_4820_ap_0_2448_epoch77_WithClincal_05-20-2022 22-28-30_CXR_Clinical]
Best AR validation model has been saved to: [val_ar_0_6366_ap_0_2947_test_ar_0_4960_ap_0_2585_epoch38_WithClincal_05-20-2022 20-47-26_CXR_Clinical]
The final model has been saved to: [val_ar_0_3336_ap_0_2024_test_ar_0_3826_ap_0_2336_epoch100_WithClincal_05-20-2022 23-30-00_CXR_Clinical]

===========================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 64,641,444
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,350,558
[model.roi_heads.box_head]: 3,342,848
[model.roi_heads.box_head.fc6]: 3,277,056
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 26,799,296
```
<img width="493" alt="image" src="https://user-images.githubusercontent.com/37566901/169541632-2a933013-dee1-4145-9a3f-49d720664380.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/169541660-0feeaee3-0ba8-4c9b-9978-56d8062b0ca0.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/169541684-65151fb0-db90-490a-ac8a-0fb7cc08a397.png">

## using only spatialisation. (0.3092, 0.2106)

```
========================================For Training [CXR_Clinical]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=256, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=30)
===========================================================================================================

Best AP validation model has been saved to: [val_ar_0_4881_ap_0_3092_test_ar_0_4164_ap_0_1880_epoch41_WithClincal_05-21-2022 03-06-53_CXR_Clinical]
Best AR validation model has been saved to: [val_ar_0_5724_ap_0_2414_test_ar_0_4497_ap_0_2106_epoch27_WithClincal_05-21-2022 02-27-22_CXR_Clinical]
The final model has been saved to: [val_ar_0_2777_ap_0_1814_test_ar_0_2741_ap_0_1469_epoch100_WithClincal_05-21-2022 05-48-05_CXR_Clinical]

===========================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 64,575,908
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,285,022
[model.roi_heads.box_head]: 3,277,312
[model.roi_heads.box_head.fc6]: 3,211,520
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 26,799,296
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/169630152-76ca4956-bc79-4acf-9c82-094d73e87945.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/169630159-3dda442e-e45d-4b48-ac57-d36c5d185250.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/169630161-54b2b6b1-1d64-4294-b828-be74aa48f7e6.png">




## using nothing. (0.2628, 0.2471)

```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=256, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=30)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_4858_ap_0_2628_test_ar_0_4535_ap_0_2471_epoch38_WithoutClincal_05-21-2022 07-03-56_CXR]
Best AR validation model has been saved to: [val_ar_0_5847_ap_0_2242_test_ar_0_6259_ap_0_2324_epoch17_WithoutClincal_05-21-2022 06-24-11_CXR]
The final model has been saved to: [val_ar_0_2778_ap_0_1871_test_ar_0_3078_ap_0_1612_epoch100_WithoutClincal_05-21-2022 08-58-57_CXR]

==================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Not using pretrained MaksRCNN model.
[model]: 30,678,253
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,285,022
[model.roi_heads.box_head]: 3,277,312
[model.roi_heads.box_head.fc6]: 3,211,520
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
```

<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/169630166-4fc6834e-f20e-40d6-8a5a-14923cf50427.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/169630174-ade49058-2771-41c3-96ec-b1c789464a41.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/169630178-79bb8145-04a0-47ec-8d0a-a3371bb2b131.png">



```

(0.2990, 0.1800)
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=256, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=30)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_5338_ap_0_2990_test_ar_0_4015_ap_0_1800_epoch28_WithoutClincal_05-21-2022 12-53-58_CXR]
Best AR validation model has been saved to: [val_ar_0_5767_ap_0_2500_test_ar_0_5245_ap_0_2210_epoch39_WithoutClincal_05-21-2022 13-13-24_CXR]
The final model has been saved to: [val_ar_0_3817_ap_0_2250_test_ar_0_3467_ap_0_2117_epoch100_WithoutClincal_05-21-2022 14-59-07_CXR]

==================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Not using pretrained MaksRCNN model.
[model]: 30,678,253
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,285,022
[model.roi_heads.box_head]: 3,277,312
[model.roi_heads.box_head.fc6]: 3,211,520
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
```

<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/169639875-ebdd1447-4f2b-429f-8acf-c976fff70fb4.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/169639881-37253cbf-b0cd-4388-8d89-37baccdff3dc.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/169639887-7aeb272e-d4ab-4750-b6d6-1b89742c1430.png">

# Larger models

## CXR_Clinical_roi_heads (0.3358, 0.2773)

```
========================================For Training [CXR_Clinical_roi_heads]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_roi_heads', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=512, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
=====================================================================================================================

Best AP validation model has been saved to: [val_ar_0_4725_ap_0_3358_test_ar_0_4604_ap_0_2773_epoch48_WithClincal_05-21-2022 19-21-28_CXR_Clinical_roi_heads]
Best AR validation model has been saved to: [val_ar_0_6042_ap_0_2981_test_ar_0_5367_ap_0_2735_epoch35_WithClincal_05-21-2022 18-57-06_CXR_Clinical_roi_heads]
The final model has been saved to: [val_ar_0_2425_ap_0_1564_test_ar_0_2705_ap_0_1891_epoch100_WithClincal_05-21-2022 20-58-41_CXR_Clinical_roi_heads]

=====================================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 57,609,627
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,416,094
[model.roi_heads.box_head]: 3,408,384
[model.roi_heads.box_head.fc6]: 3,342,592
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 26,799,296
```


## CXR_Clinical_roi_heads_spatialisation (0.3054, 0.2220)

```
========================================For Training [CXR_Clinical_roi_heads_spatialisation]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_roi_heads_spatialisation', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=512, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
====================================================================================================================================

Best AP validation model has been saved to: [val_ar_0_6252_ap_0_3054_test_ar_0_4563_ap_0_2220_epoch21_WithClincal_05-21-2022 22-02-19_CXR_Clinical_roi_heads_spatialisation]
Best AR validation model has been saved to: [val_ar_0_6277_ap_0_2452_test_ar_0_5815_ap_0_3068_epoch33_WithClincal_05-21-2022 22-36-19_CXR_Clinical_roi_heads_spatialisation]
The final model has been saved to: [val_ar_0_3824_ap_0_1961_test_ar_0_3223_ap_0_1831_epoch100_WithClincal_05-22-2022 01-42-46_CXR_Clinical_roi_heads_spatialisation]

====================================================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 64,969,636
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,416,094
[model.roi_heads.box_head]: 3,408,384
[model.roi_heads.box_head.fc6]: 3,342,592
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 26,799,296
```

## CXR_Clinical_spatialisation (0.3134, 0.2311)
```
========================================For Training [CXR_Clinical_spatialisation]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_spatialisation', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=512, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
==========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_6109_ap_0_3134_test_ar_0_4928_ap_0_2311_epoch42_WithClincal_05-22-2022 03-45-41_CXR_Clinical_spatialisation]
Best AR validation model has been saved to: [val_ar_0_6109_ap_0_3134_test_ar_0_4928_ap_0_2311_epoch42_WithClincal_05-22-2022 03-45-40_CXR_Clinical_spatialisation]
The final model has been saved to: [val_ar_0_3833_ap_0_2185_test_ar_0_3537_ap_0_2130_epoch100_WithClincal_05-22-2022 06-27-35_CXR_Clinical_spatialisation]

==========================================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 64,838,564
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,285,022
[model.roi_heads.box_head]: 3,277,312
[model.roi_heads.box_head.fc6]: 3,211,520
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 26,799,296
```


## CXR (0.3149, 0.2402)

```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler=None, reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=10, reduceLROnPlateau_full_stop=False, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=512, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_6518_ap_0_3149_test_ar_0_4803_ap_0_2402_epoch55_WithoutClincal_05-22-2022 08-18-06_CXR]
Best AR validation model has been saved to: [val_ar_0_6518_ap_0_3149_test_ar_0_4803_ap_0_2402_epoch55_WithoutClincal_05-22-2022 08-18-06_CXR]
The final model has been saved to: [val_ar_0_4440_ap_0_1848_test_ar_0_4305_ap_0_2164_epoch100_WithoutClincal_05-22-2022 09-43-33_CXR]

==================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Not using pretrained MaksRCNN model.
[model]: 30,678,253
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,285,022
[model.roi_heads.box_head]: 3,277,312
[model.roi_heads.box_head.fc6]: 3,211,520
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
```

# Newer attempts

### CXR_Clinical_roi_heads_spatialisation (0.3474, 0.1862)

```
========================================For Training [CXR_Clinical_roi_heads_spatialisation]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_roi_heads_spatialisation', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=512, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=0)
====================================================================================================================================

Best AP validation model has been saved to: [val_ar_0_4536_ap_0_3474_test_ar_0_2630_ap_0_1862_epoch61_WithClincal_05-23-2022 04-51-20_CXR_Clinical_roi_heads_spatialisation]
Best AR validation model has been saved to: [val_ar_0_5623_ap_0_3135_test_ar_0_3942_ap_0_1994_epoch35_WithClincal_05-23-2022 03-38-21_CXR_Clinical_roi_heads_spatialisation]
The final model has been saved to: [val_ar_0_3982_ap_0_2418_test_ar_0_2669_ap_0_1925_epoch100_WithClincal_05-23-2022 06-40-14_CXR_Clinical_roi_heads_spatialisation]

====================================================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 64,969,636
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,416,094
[model.roi_heads.box_head]: 3,408,384
[model.roi_heads.box_head.fc6]: 3,342,592
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 26,799,296
```

## CXR_Clinical_roi_heads (0.3146, 0.2546)

```
========================================For Training [CXR_Clinical_roi_heads]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_roi_heads', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='unified', image_size=512, backbone_out_channels=16, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=[100], multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=512, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=0)
=====================================================================================================================

Best AP validation model has been saved to: [val_ar_0_6366_ap_0_3146_test_ar_0_5370_ap_0_2546_epoch38_WithClincal_05-23-2022 07-57-51_CXR_Clinical_roi_heads]
Best AR validation model has been saved to: [val_ar_0_6366_ap_0_3146_test_ar_0_5370_ap_0_2546_epoch38_WithClincal_05-23-2022 07-57-50_CXR_Clinical_roi_heads]
The final model has been saved to: [val_ar_0_4022_ap_0_1978_test_ar_0_3231_ap_0_1923_epoch100_WithClincal_05-23-2022 09-52-28_CXR_Clinical_roi_heads]

=====================================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 57,609,627
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,416,094
[model.roi_heads.box_head]: 3,408,384
[model.roi_heads.box_head.fc6]: 3,342,592
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 26,799,296
```


# Small model 

## MobileNet v3 (0.1911, 0.3168)
```
========================================For Training [CXR_Clinical_roi_heads_spatialisation]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_roi_heads_spatialisation', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
====================================================================================================================================

Best AP validation model has been saved to: [val_ar_0_4491_ap_0_2154_test_ar_0_4870_ap_0_2487_epoch48_WithClincal_05-23-2022 12-04-14_CXR_Clinical_roi_heads_spatialisation]
Best AR validation model has been saved to: [val_ar_0_5436_ap_0_1911_test_ar_0_5476_ap_0_3168_epoch49_WithClincal_05-23-2022 12-06-22_CXR_Clinical_roi_heads_spatialisation]
The final model has been saved to: [val_ar_0_2797_ap_0_1456_test_ar_0_3159_ap_0_1591_epoch100_WithClincal_05-23-2022 13-43-05_CXR_Clinical_roi_heads_spatialisation]

====================================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Using pretrained backbone. mobilenet_v3
[model]: 3,217,952
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 210,974
[model.roi_heads.box_head]: 209,024
[model.roi_heads.box_head.fc6]: 204,864
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.clinical_convs]: 1,258,848
[model.fuse_convs]: 0
```

<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/169771740-b2cbe62d-a9b1-4f3a-ad08-ac52dbb5215b.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/169771765-a8471986-6dbd-42d0-aa99-7ca7fb8be376.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/169771788-2266fdd2-6551-418c-9044-55659ef887de.png">

## ResNet50 (0.1971, 0.2651)
```
========================================For Training [CXR_Clinical_roi_heads_spatialisation]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_roi_heads_spatialisation', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
====================================================================================================================================

Best AP validation model has been saved to: [val_ar_0_5126_ap_0_2309_test_ar_0_4089_ap_0_1735_epoch61_WithClincal_05-23-2022 16-17-19_CXR_Clinical_roi_heads_spatialisation]
Best AR validation model has been saved to: [val_ar_0_5657_ap_0_1971_test_ar_0_6117_ap_0_2651_epoch35_WithClincal_05-23-2022 15-15-17_CXR_Clinical_roi_heads_spatialisation]
The final model has been saved to: [val_ar_0_3561_ap_0_1923_test_ar_0_3517_ap_0_1855_epoch100_WithClincal_05-23-2022 17-50-54_CXR_Clinical_roi_heads_spatialisation]

====================================================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 55,453,092
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 813,086
[model.roi_heads.box_head]: 811,136
[model.roi_heads.box_head.fc6]: 806,976
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.clinical_convs]: 26,799,296
```

<img width="501" alt="image" src="https://user-images.githubusercontent.com/37566901/169771867-b25003f7-3ff5-4b9c-b8d1-dab0df695a44.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/169771901-462a7995-d47a-4ce9-9f1c-78bdd4da460c.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/169771925-74e91ce8-4bca-4200-a837-400c981462d6.png">


# Large model


## mobilenet_v3 (0.2218, 0.2661)
```
========================================For Training [CXR_Clinical_roi_heads_spatialisation]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_roi_heads_spatialisation', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=256, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=256, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
====================================================================================================================================

Best AP validation model has been saved to: [val_ar_0_4209_ap_0_2218_test_ar_0_4825_ap_0_2661_epoch46_WithClincal_05-23-2022 20-29-03_CXR_Clinical_roi_heads_spatialisation]
Best AR validation model has been saved to: [val_ar_0_6008_ap_0_1958_test_ar_0_6498_ap_0_2531_epoch44_WithClincal_05-23-2022 20-24-17_CXR_Clinical_roi_heads_spatialisation]
The final model has been saved to: [val_ar_0_3393_ap_0_1440_test_ar_0_3933_ap_0_1990_epoch100_WithClincal_05-23-2022 22-32-50_CXR_Clinical_roi_heads_spatialisation]

====================================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Using pretrained backbone. mobilenet_v3
[model]: 15,567,008
[model.backbone]: 2,254,368
[model.rpn]: 609,355
[model.roi_heads]: 3,350,558
[model.roi_heads.box_head]: 3,342,848
[model.roi_heads.box_head.fc6]: 3,277,056
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 2,254,368
[model.fuse_convs]: 0
```
<img width="499" alt="image" src="https://user-images.githubusercontent.com/37566901/169925078-a5b6d5be-597f-4142-8b3c-2b0503398e5c.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/169925091-9ccd396d-c31c-4d97-b67c-64c6a686a8e5.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/169925102-4b77cbc2-9674-4831-a247-2d4a6d5f1ece.png">



## ResNet50 (0.1710, 0.2330)
```
========================================For Training [CXR_Clinical_roi_heads_spatialisation]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_roi_heads_spatialisation', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='resnet50', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=256, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=256, mask_hidden_layers=256, using_fpn=True, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=256, clinical_expand_conv_channels=256, clinical_num_len=9, clinical_conv_channels=256, fuse_conv_channels=32, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
====================================================================================================================================

Best AP validation model has been saved to: [val_ar_0_5086_ap_0_2506_test_ar_0_5187_ap_0_2209_epoch44_WithClincal_05-24-2022 00-46-48_CXR_Clinical_roi_heads_spatialisation]
Best AR validation model has been saved to: [val_ar_0_5407_ap_0_1710_test_ar_0_5937_ap_0_2330_epoch72_WithClincal_05-24-2022 02-07-53_CXR_Clinical_roi_heads_spatialisation]
The final model has been saved to: [val_ar_0_3339_ap_0_1738_test_ar_0_3111_ap_0_1893_epoch100_WithClincal_05-24-2022 03-28-36_CXR_Clinical_roi_heads_spatialisation]

====================================================================================================================================
Load custom model
Using ResNet as backbone
Using pretrained backbone. resnet50
Using ResNet as clinical backbone
Not using pretrained MaksRCNN model.
[model]: 64,641,444
[model.backbone]: 26,799,296
[model.rpn]: 593,935
[model.roi_heads]: 3,350,558
[model.roi_heads.box_head]: 3,342,848
[model.roi_heads.box_head.fc6]: 3,277,056
[model.roi_heads.box_head.fc7]: 65,792
[model.roi_heads.box_predictor]: 7,710
[model.clinical_convs]: 26,799,296
```

<img width="493" alt="image" src="https://user-images.githubusercontent.com/37566901/169925136-1e8dbc0d-20f9-41ec-8ed8-0308c7daf35e.png">
<img width="515" alt="image" src="https://user-images.githubusercontent.com/37566901/169925144-7e6b3243-b041-4749-9f73-7ae58215952b.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/169925154-0ecfef7c-b9ae-4c9f-ab31-426081d1e5e8.png">


# Small CXR model (0.1785, 0.2089)
```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_5013_ap_0_2228_test_ar_0_4743_ap_0_2090_epoch42_WithoutClincal_05-24-2022 12-22-36_CXR]
Best AR validation model has been saved to: [val_ar_0_5271_ap_0_1785_test_ar_0_5737_ap_0_2089_epoch45_WithoutClincal_05-24-2022 12-27-53_CXR]
The final model has been saved to: [val_ar_0_3589_ap_0_1613_test_ar_0_3565_ap_0_1762_epoch100_WithoutClincal_05-24-2022 13-57-31_CXR]

==================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
[model]: 1,507,529
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 206,878
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
```
<img width="331" alt="image" src="https://user-images.githubusercontent.com/37566901/169956595-9d15da5b-a7bf-4134-a6c3-9895f1c33249.png">
<img width="346" alt="image" src="https://user-images.githubusercontent.com/37566901/169956612-c1f7a28e-f3aa-4e34-b91c-67440de8e2f3.png">
<img width="346" alt="image" src="https://user-images.githubusercontent.com/37566901/169956635-bf97236a-f8e9-40b9-9170-937681c42437.png">

# Ablation:

## CXR_Clinical_fusion1 (0.2017, 0.2767)
```
========================================For Training [CXR_Clinical_fusion1]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_fusion1', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999)
===================================================================================================================

Best AP validation model has been saved to: [val_ar_0_4854_ap_0_2239_test_ar_0_5159_ap_0_2580_epoch44_WithClincal_05-25-2022 01-48-27_CXR_Clinical_fusion1]
Best AR validation model has been saved to: [val_ar_0_5353_ap_0_2017_test_ar_0_6190_ap_0_2767_epoch45_WithClincal_05-25-2022 01-50-35_CXR_Clinical_fusion1]
The final model has been saved to: [val_ar_0_3959_ap_0_1863_test_ar_0_3997_ap_0_1865_epoch100_WithClincal_05-25-2022 03-32-06_CXR_Clinical_fusion1]

===================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Using pretrained backbone. mobilenet_v3
[model]: 3,213,856
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 206,878
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.clinical_convs]: 1,258,848
[model.fuse_convs]: 0
```

## CXR_Clinical_fusion2 (0.2274, 0.2175)

```
========================================For Training [CXR_Clinical_fusion2]======================================== 
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_fusion2', best_ar_val_model_path=None, best_ap_val_model_path=None, final_model_path=None, backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=False, clinical_expand_dropout_rate=0, clinical_conv_dropout_rate=0, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, fuse_dropout_rate=0, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999) =================================================================================================================== 
Best AP validation model has been saved to: [val_ar_0_4654_ap_0_2274_test_ar_0_4886_ap_0_2157_epoch85_WithClincal_05-25-2022 06-02-35_CXR_Clinical_fusion2] 
Best AR validation model has been saved to: [val_ar_0_5760_ap_0_1744_test_ar_0_5622_ap_0_1990_epoch49_WithClincal_05-25-2022 05-01-33_CXR_Clinical_fusion2] 
The final model has been saved to: [val_ar_0_2985_ap_0_1360_test_ar_0_2752_ap_0_1522_epoch100_WithClincal_05-25-2022 06-27-43_CXR_Clinical_fusion2] =================================================================================================================== 
Load custom model Using pretrained backbone. 
mobilenet_v3 Using pretrained backbone.
mobilenet_v3 
[model]: 2,770,583 
[model.backbone]: 1,258,848 
[model.rpn]: 41,803 
[model.roi_heads]: 210,974 
[model.roi_heads.box_head]: 209,024 
[model.roi_heads.box_head.fc6]: 204,864 
[model.roi_heads.box_head.fc7]: 4,160 
[model.roi_heads.box_predictor]: 1,950 
[model.clinical_convs]: 1,258,848
```
### Ablation 2:

## CXR_Clinical_fusion1_fusino2 (0.2886)
```
========================================For Training [CXR_Clinical_fusion1_fusino2]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_fusion1_fusino2', backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999, spatialise_method='convs', normalise_clinical_num=False, measure_test=True)
===========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_5671_ap_0_2109_test_ar_0_6467_ap_0_2886_epoch57_WithClincal_05-29-2022 01-12-26_CXR_Clinical_fusion1_fusino2]
Best AR validation model has been saved to: [val_ar_0_5671_ap_0_2109_test_ar_0_6467_ap_0_2886_epoch57_WithClincal_05-29-2022 01-12-25_CXR_Clinical_fusion1_fusino2]
The final model has been saved to: [val_ar_0_4104_ap_0_1642_test_ar_0_4676_ap_0_2482_epoch80_WithClincal_05-29-2022 02-24-52_CXR_Clinical_fusion1_fusino2]

===========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 5,399,846
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,392,868
[model.roi_heads.box_head]: 209,024
[model.roi_heads.box_head.fc6]: 204,864
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
[model.clinical_convs]: 1,258,848
Max AP on test: [0.3450]
```
<img width="499" alt="image" src="https://user-images.githubusercontent.com/37566901/170847004-820f7d43-ee20-4ee7-ba2e-c5ae45689965.png">
<img width="515" alt="image" src="https://user-images.githubusercontent.com/37566901/170847008-d411c2cc-e931-42fa-9d32-2e9f18917ad5.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/170847013-8d13633a-609f-4f0c-852c-09e71260241f.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/170847015-cb444006-0a90-4734-8405-ed3f030e4389.png">

## CXR_Clinical_fusion1 (0.2382)
```
========================================For Training [CXR_Clinical_fusion1]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_fusion1', backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999, spatialise_method='convs', normalise_clinical_num=False, measure_test=True)
===================================================================================================================

Best AP validation model has been saved to: [val_ar_0_4441_ap_0_2097_test_ar_0_4911_ap_0_2267_epoch35_WithClincal_05-29-2022 04-11-27_CXR_Clinical_fusion1]
Best AR validation model has been saved to: [val_ar_0_5293_ap_0_1752_test_ar_0_5978_ap_0_2382_epoch37_WithClincal_05-29-2022 04-18-42_CXR_Clinical_fusion1]
The final model has been saved to: [val_ar_0_2438_ap_0_1380_test_ar_0_3197_ap_0_1619_epoch80_WithClincal_05-29-2022 06-27-44_CXR_Clinical_fusion1]

===================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 5,395,750
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,388,772
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
[model.clinical_convs]: 1,258,848
Max AP on test: [0.2765]
```

<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/170847021-05fbf154-bb75-4785-b798-7eed56788739.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/170847053-223f185d-d3ad-42e3-946b-a707f3f04153.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/170847056-2a4239f5-7eba-4fee-b2e6-c745ee583e63.png">
<img width="522" alt="image" src="https://user-images.githubusercontent.com/37566901/170847058-a3cf400f-a6ab-4e0b-a3be-eea3da46d0c7.png">


## CXR_Clinical_fusion2 (0.2222)

```
========================================For Training [CXR_Clinical_fusion2]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_fusion2', backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999, spatialise_method='convs', normalise_clinical_num=False, measure_test=True)
===================================================================================================================

Best AP validation model has been saved to: [val_ar_0_4660_ap_0_2135_test_ar_0_4463_ap_0_1989_epoch31_WithClincal_05-29-2022 07-47-46_CXR_Clinical_fusion2]
Best AR validation model has been saved to: [val_ar_0_5598_ap_0_1760_test_ar_0_5254_ap_0_2222_epoch62_WithClincal_05-29-2022 09-21-49_CXR_Clinical_fusion2]
The final model has been saved to: [val_ar_0_3787_ap_0_1570_test_ar_0_3762_ap_0_2451_epoch80_WithClincal_05-29-2022 10-14-16_CXR_Clinical_fusion2]

===================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 3,693,629
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,392,868
[model.roi_heads.box_head]: 209,024
[model.roi_heads.box_head.fc6]: 204,864
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.3018]
```
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/170847105-452c6959-2d23-4591-b8dd-4a4fbee083de.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/170847110-b5ec8959-0357-487d-a4a2-0c214c1dfe1e.png">
![image](https://user-images.githubusercontent.com/37566901/170847114-f3c5cf12-2fe3-4148-93f4-d7fe2a95c38e.png)

## CXR (0.1961)

```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999, spatialise_method='convs', normalise_clinical_num=False, measure_test=True)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_4483_ap_0_2048_test_ar_0_5114_ap_0_1746_epoch52_WithoutClincal_05-29-2022 13-18-59_CXR]
Best AR validation model has been saved to: [val_ar_0_5659_ap_0_1741_test_ar_0_5390_ap_0_1961_epoch36_WithoutClincal_05-29-2022 12-29-51_CXR]
The final model has been saved to: [val_ar_0_3937_ap_0_1574_test_ar_0_3679_ap_0_1713_epoch60_WithoutClincal_05-29-2022 13-43-13_CXR]

==================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 3,689,423
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,388,772
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.2428]
```

<img width="498" alt="image" src="https://user-images.githubusercontent.com/37566901/170852153-ef410114-2b8a-42ae-8464-c523db559bb8.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/170852157-fd94250e-7ef8-4983-9f32-53f871122187.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/170852163-ad36915f-a681-4baf-88fc-8ac5341a7f02.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/170852165-7f42c880-3b43-4e52-b367-2ee303f3c7bc.png">


# last attempts without normalisation

## CXR_Clinical_fusion1_fusino2 (0.2800)

```
========================================For Training [CXR_Clinical_fusion1_fusino2]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_fusion1_fusino2', backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999, spatialise_method='convs', normalise_clinical_num=False, measure_test=True)
===========================================================================================================================

Best AP validation model has been saved to: [val_ar_0_4663_ap_0_2049_test_ar_0_4581_ap_0_2607_epoch89_WithClincal_05-30-2022 05-07-31_CXR_Clinical_fusion1_fusino2]
Best AR validation model has been saved to: [val_ar_0_5772_ap_0_1653_test_ar_0_6333_ap_0_2800_epoch57_WithClincal_05-30-2022 03-24-24_CXR_Clinical_fusion1_fusino2]
The final model has been saved to: [val_ar_0_2753_ap_0_1182_test_ar_0_2822_ap_0_1332_epoch100_WithClincal_05-30-2022 05-41-15_CXR_Clinical_fusion1_fusino2]

===========================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 5,399,846
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,392,868
[model.roi_heads.box_head]: 209,024
[model.roi_heads.box_head.fc6]: 204,864
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
[model.clinical_convs]: 1,258,848
Max AP on test: [0.2994]
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/171006786-ca68a017-2214-45e4-80fe-d4a20a649cc7.png">
<img width="525" alt="image" src="https://user-images.githubusercontent.com/37566901/171006803-4db20477-cb84-41d8-973e-0681d59ebd71.png">
<img width="521" alt="image" src="https://user-images.githubusercontent.com/37566901/171006824-a9beb8c5-c8e3-41aa-bff4-06fc425e9486.png">
<img width="516" alt="image" src="https://user-images.githubusercontent.com/37566901/171006842-405204bd-778d-45ba-9cd5-bf334b8e75a4.png">


## CXR_Clinical_fusion1 (0.2757)

```
========================================For Training [CXR_Clinical_fusion1]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_fusion1', backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=True, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999, spatialise_method='convs', normalise_clinical_num=False, measure_test=True)
===================================================================================================================

Best AP validation model has been saved to: [val_ar_0_5292_ap_0_2123_test_ar_0_5660_ap_0_2347_epoch43_WithClincal_05-30-2022 08-09-09_CXR_Clinical_fusion1]
Best AR validation model has been saved to: [val_ar_0_5476_ap_0_1984_test_ar_0_6038_ap_0_2757_epoch41_WithClincal_05-30-2022 08-01-54_CXR_Clinical_fusion1]
The final model has been saved to: [val_ar_0_3172_ap_0_1321_test_ar_0_3740_ap_0_2147_epoch100_WithClincal_05-30-2022 11-10-35_CXR_Clinical_fusion1]

===================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 5,395,750
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,388,772
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
[model.clinical_convs]: 1,258,848
Max AP on test: [0.2870]
```
<img width="497" alt="image" src="https://user-images.githubusercontent.com/37566901/171006910-c7c0cd1b-d811-4105-943c-8d9121fd4024.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/171006930-87a67532-7f1e-4be0-92e2-fa63d698c32e.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/171006959-b02d2186-9565-41ac-bebc-c3e8b09d96f3.png">
<img width="520" alt="image" src="https://user-images.githubusercontent.com/37566901/171006974-1531c549-ba0e-4a68-8787-29e31e7d9f91.png">


## CXR_Clinical_fusion2 (0.2218)

```
========================================For Training [CXR_Clinical_fusion2]========================================
ModelSetup(use_clinical=True, use_custom_model=True, use_early_stop_model=True, name='CXR_Clinical_fusion2', backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=True, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999, spatialise_method='convs', normalise_clinical_num=False, measure_test=True)
===================================================================================================================

Best AP validation model has been saved to: [val_ar_0_4369_ap_0_2098_test_ar_0_4940_ap_0_2218_epoch58_WithClincal_05-30-2022 13-58-43_CXR_Clinical_fusion2]
Best AR validation model has been saved to: [val_ar_0_5746_ap_0_1881_test_ar_0_5930_ap_0_1940_epoch36_WithClincal_05-30-2022 12-50-44_CXR_Clinical_fusion2]
The final model has been saved to: [val_ar_0_2983_ap_0_1102_test_ar_0_4190_ap_0_2044_epoch100_WithClincal_05-30-2022 15-56-16_CXR_Clinical_fusion2]

===================================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 3,693,629
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,392,868
[model.roi_heads.box_head]: 209,024
[model.roi_heads.box_head.fc6]: 204,864
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.2713]
```
<img width="496" alt="image" src="https://user-images.githubusercontent.com/37566901/171007060-053db99c-b7cb-4e52-9d80-fbcce7fde8b0.png">
<img width="519" alt="image" src="https://user-images.githubusercontent.com/37566901/171007082-ffc39668-9151-4edd-9a22-a4abf8f41366.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/171007103-6a25303e-926e-47c8-8e3f-904a9904cc4a.png">
<img width="521" alt="image" src="https://user-images.githubusercontent.com/37566901/171007119-dcd27623-1461-4f3f-bea7-2934973239b9.png">

## CXR (0.2211)

```
========================================For Training [CXR]========================================
ModelSetup(use_clinical=False, use_custom_model=True, use_early_stop_model=True, name='CXR', backbone='mobilenet_v3', optimiser='sgd', lr=0.001, weight_decay=0, pretrained=True, record_training_performance=True, dataset_mode='normal', image_size=512, backbone_out_channels=64, batch_size=4, warmup_epochs=0, lr_scheduler='ReduceLROnPlateau', reduceLROnPlateau_factor=0.1, reduceLROnPlateau_patience=999, reduceLROnPlateau_full_stop=True, multiStepLR_milestones=100, multiStepLR_gamma=0.1, representation_size=64, mask_hidden_layers=64, using_fpn=False, use_mask=True, clinical_input_channels=64, clinical_expand_conv_channels=64, clinical_num_len=9, clinical_conv_channels=64, fuse_conv_channels=64, box_head_dropout_rate=0, fuse_depth=0, spatialise_clinical=False, add_clinical_to_roi_heads=False, fusion_strategy='add', fusion_residule=False, gt_in_train_till=999, spatialise_method='convs', normalise_clinical_num=False, measure_test=True)
==================================================================================================

Best AP validation model has been saved to: [val_ar_0_5306_ap_0_2082_test_ar_0_5610_ap_0_2211_epoch73_WithoutClincal_05-30-2022 19-31-22_CXR]
Best AR validation model has been saved to: [val_ar_0_5306_ap_0_2082_test_ar_0_5610_ap_0_2211_epoch73_WithoutClincal_05-30-2022 19-31-22_CXR]
The final model has been saved to: [val_ar_0_2997_ap_0_1580_test_ar_0_3127_ap_0_1845_epoch100_WithoutClincal_05-30-2022 20-42-07_CXR]

==================================================================================================
Load custom model
Using pretrained backbone. mobilenet_v3
Mask Hidden Layers 256
[model]: 3,689,423
[model.backbone]: 1,258,848
[model.rpn]: 41,803
[model.roi_heads]: 2,388,772
[model.roi_heads.box_head]: 204,928
[model.roi_heads.box_head.fc6]: 200,768
[model.roi_heads.box_head.fc7]: 4,160
[model.roi_heads.box_predictor]: 1,950
[model.roi_heads.mask_head]: 1,917,952
Max AP on test: [0.2503]
```
<img width="496" alt="image" src="https://user-images.githubusercontent.com/37566901/171007241-ba4d95a6-c702-4fe3-9ec0-153a2fa5d0be.png">
<img width="515" alt="image" src="https://user-images.githubusercontent.com/37566901/171007268-582e32e6-1949-498b-b9b2-ee43b1e595ff.png">
<img width="517" alt="image" src="https://user-images.githubusercontent.com/37566901/171007292-96b593db-5bfa-4f5b-9e4f-e64915bcd22e.png">
<img width="518" alt="image" src="https://user-images.githubusercontent.com/37566901/171007304-ef6f14d9-5fc3-4395-821c-aef0255269a4.png">

### Improvement attempts,

## Attemps of evaluation:

Q: When we're trying to evaluate the model, the CXR + clinical model does has a better performance on the validation and test sets. However, when we actually print out the bounding boxes, the CXR model seems to make more sense.

E1: Since we usually use 0.3 as the threshold for generating the bounding boxes, we can apply the same threshold for evaluation and see what's the performance gap between default (thrs=0.3) and thrs=0.3.

L1: From the table below, we can see the gap between thrs=0.3 and thrs=0.05, and the CXR+Clinical(lr=1e-2) has a big performance drop when it get thrs=0.3.

|   |CXR|CXR+Clinical(lr=1e-3)|CXR+Clinical(lr=1e-2)|
|---|---|---|---|
|score_thrs=0.05|0.2716|0.4956|0.5391|
|score_thrs=0.1|0.2614|0.4904|0.4870|
|score_thrs=0.3|0.1475|0.3451|0.2468|

E2: We will also need to check if the CXR + clinical model has a better performance at thrs=0.3.
L2: As the model (lr=1e-3) still has a better performance. The model (lr=1e-2) has a large performance gap.

E3: the original CXR + clinical model may have even more performance drop.
L3: Yes, the orignial one is far worse than the lr=1e-3 one.

E4: What if we use add as the fusion strategy?
L4: It works similar to concatenation.

E5: what if we make the fusion strategy residule?
L5: If we use residule in fusion, then the training graph, AP and AR become more like the CXR model.

E6: Why the performance is such low comparing to the ap and ar we got during training.
L6: False issue, it's just a wrong coco api fed into the evaluator.
 
### This is pretty close to the regulation limit.

```
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
        weight_decay=1e-3,
        record_training_performance=True,
        using_fpn=False,
        backbone_out_channels=16,  # shrink size test [32]
        representation_size=64,  # shrink size test [128]
        mask_hidden_layers=256,
        use_mask=False,
    ),
```


