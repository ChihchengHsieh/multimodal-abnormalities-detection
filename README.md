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
### With dropout=0.2 in box_head.

<img width="398" alt="image" src="https://user-images.githubusercontent.com/37566901/165445268-14a727ef-751d-42e4-8638-38e9ea2b71ad.png">

### 

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


