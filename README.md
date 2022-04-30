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





#### Baseline and comparisons

### dropout=0, with ReduceLROnPlateau, factor=0.1, patience=3 (this is the baseline with pretty decent result) (v=0.3130, t=0.3557)

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

### dropout=0, with MultiStepLR, multiStepLR_milestones=[30, 50, 70, 90] (v=0.2944, t=0.3029)

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

### dropout=0, with ReduceLROnPlateau, factor=0.1, patience=3, slightly larger model. (v=0.2996, t=0.2971)

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

### dropout=0.2, with ReduceLROnPlateau, factor=0.1, patience=3 (0.751) (it should have some warmup epochs that I forgot to set.)
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


### dropout=0, with ReduceLROnPlateau, factor=0.5, patience=2 (v=0.3257, t=0.3381)
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


### dropout=0, with MultiStepLR, multiStepLR_milestones=[10 , 30, 50, 70, 90], multiStepLR_gamma=0.1 (v=0.3469, t=0.2877)

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


### dropout=0, with ReduceLROnPlateau, factor=0.1, patience=2 (v=0.3210, t=0.3096)
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


### dropout=0.2, with ReduceLROnPlateau, factor=0.1, patience=3 (v=0.2702, t=0.3200)

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



### 1 baseline, clinical
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


### 2 MultiStepLR, clinical
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


### 3 larger, clinical

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

### 4 doprout = 0.2, clinincal

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


