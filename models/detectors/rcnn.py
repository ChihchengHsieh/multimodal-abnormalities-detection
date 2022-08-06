import torch
import warnings
import itertools
from collections import OrderedDict
from torch import nn, Tensor
from typing import Tuple, List, Dict

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNNHeads
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.faster_rcnn import (
    MultiScaleRoIAlign,
    RPNHead,
    FastRCNNPredictor,
    AnchorGenerator,
    GeneralizedRCNNTransform,
)

from torchvision.models.detection.transform import resize_boxes, resize_keypoints
from torchvision.models.detection.roi_heads import paste_masks_in_image
from models.detectors.rcnn_components import (
    XAMIRegionProposalNetwork,
    XAMIRoIHeads,
    XAMITwoMLPHead,
)

from models.setup import ModelSetup


class MultimodalGeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(
        self,
        setup: ModelSetup,
        backbone,
        rpn,
        roi_heads,
        transform,
        fixation_backbone=None,
    ):

        super(MultimodalGeneralizedRCNN, self).__init__()

        self.transform = transform
        self.backbone = backbone
        self.fixation_backbone = fixation_backbone

        self.backbone_output_channels = self.backbone.out_channels
        self.rpn = rpn
        self.roi_heads = roi_heads

        self.setup = setup

        example_img_features = self.backbone(
            self.transform([torch.ones(3, 2048, 2048)])[0].tensors
        )

        # if isinstance(example_img_features, OrderedDict):
        if isinstance(example_img_features, torch.Tensor):
            self.feature_keys = ["0"]
            example_img_features = OrderedDict([("0", example_img_features)])
        elif isinstance(example_img_features, OrderedDict):
            self.feature_keys = example_img_features.keys()
        else:
            raise Exception("Unsupported output format from image backbone.")

        # determine the output size for image backbone, which is the size of the image feature maps.
        last_key = list(example_img_features.keys())[-1]
        self.image_feature_map_size = example_img_features[last_key].shape[-1]

        self.build_fuse_convs()

    def build_fuse_convs(self,):
        if self.setup.using_fpn:
            self._build_fpn_fuse_convs()
        else:
            self._build_normal_fuse_convs()

    def _build_fpn_fuse_convs(self,):
        if self.setup.fuse_depth == 0:
            return

        self.fuse_convs = nn.ModuleDict({})
        for i, key in enumerate(self.feature_keys):
            network = [
                nn.Conv2d(
                    (
                        self.get_fuse_input_channel()
                        if i == 0
                        else self.backbone_output_channels
                    ),
                    self.backbone_output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(self.backbone_output_channels),
                nn.ReLU(inplace=False),
            ]
            self.fuse_convs[key] = nn.Sequential(*network)

    def _build_normal_fuse_convs(self,):
        if self.setup.fuse_depth == 0:
            return

        fuse_convs_modules = list(
            itertools.chain.from_iterable(
                [
                    [
                        nn.Conv2d(
                            (
                                self.get_fuse_input_channel()
                                if i == 0
                                else self.setup.backbone_out_channels
                            ),
                            (self.backbone_output_channels),
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(self.backbone_output_channels),
                        nn.ReLU(inplace=False),
                    ]
                    for i in range(self.setup.fuse_depth)
                ]
            )
        )

        self.fuse_convs = nn.Sequential(*fuse_convs_modules)

    def get_fuse_input_channel(self,):
        if self.setup.fusion_strategy == "concat":
            return self.backbone_output_channels * 2
        elif self.setup.fusion_strategy == "add":
            return self.backbone_output_channels
        else:
            raise Exception(
                f"Unsupported fusion strategy: {self.setup.fusion_strategy}"
            )

    def get_clinical_features(self, clinical_num, clinical_cat, img_features):
        clinical_input = None
        if self.setup.use_clinical:
            clincal_embout = self.gender_emb_layer(torch.concat(clinical_cat, axis=0))
            clinical_input = torch.concat(
                [torch.stack(clinical_num, dim=0), clincal_embout], axis=1
            )

        clinical_features = None
        if self.setup.spatialise_clinical:
            clinical_features = OrderedDict({})
            if self.setup.spatialise_method == "convs":
                clinical_expanded_input = self.clinical_expand_conv(
                    clinical_input[:, :, None, None]
                )
                self.last_clinical_expanded_input = clinical_expanded_input
                clinical_features = self.clinical_convs(clinical_expanded_input)

                if isinstance(clinical_features, torch.Tensor):
                    clinical_features = OrderedDict([("0", clinical_features)])
            elif self.setup.spatialise_method == "repeat":

                for k in self.feature_keys:
                    clinical_features[k] = self.before_repeat[k](clinical_input)[
                        :, :, None, None
                    ].repeat(
                        1, 1, img_features[k].shape[-2], img_features[k].shape[-1],
                    )
            else:
                raise Exception(
                    "Unsupported spatialise method: {self.setup.sptailise_method}"
                )

        return clinical_input, clinical_features

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        return losses, detections

    def fuse_feature_maps(
        self, img_feature: torch.Tensor, clinical_feature: torch.Tensor
    ) -> torch.Tensor:
        if self.setup.fusion_strategy == "concat":
            return torch.concat([img_feature, clinical_feature], axis=1)
        elif self.setup.fusion_strategy == "add":
            return img_feature + clinical_feature
        else:
            raise Exception(
                f"Unsupported fusion strategy: {self.setup.fusion_strategyn}"
            )

    def fuse_features(self, img_features, clinical_features):
        features = OrderedDict({})
        if self.setup.using_fpn:
            for k in self.feature_keys:
                if self.setup.fusion_strategy == "add" and self.setup.fuse_depth == 0:
                    features[k] = self.fuse_feature_maps(
                        img_features[k], clinical_features[k]
                    )
                else:
                    features[k] = self.fuse_convs[k](
                        self.fuse_feature_maps(img_features[k], clinical_features[k])
                    )

                if self.setup.fusion_strategy == "add" and self.setup.fusion_residule:
                    features[k] = features[k] + img_features[k] + clinical_features[k]

        else:
            k = "0"
            if self.setup.fusion_strategy == "add" and self.setup.fuse_depth == 0:
                features[k] = self.fuse_feature_maps(
                    img_features[k], clinical_features[k]
                )

            else:
                features[k] = self.fuse_convs(
                    self.fuse_feature_maps(img_features[k], clinical_features[k])
                )

            if self.setup.fusion_strategy == "add" and self.setup.fusion_residule:
                features[k] = features[k] + img_features[k] + clinical_features[k]

        return features

    def valid_bbox(self, targets):

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        " Found invalid box {} for target at index {}.".format(
                            degen_bb, target_idx
                        )
                    )

    def forward(
        self, images, fixations=None, targets=None
    ):

        """
        Args
            images (list[Tensor]): images to be processed
            fixations (list[Tensor]): fixations to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.setup.use_fixations:
            assert (not fixations is None) , "Expecting `fixation_masks` as input."

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            "Expected target boxes to be a tensor"
                            "of shape [N, 4], got {:}.".format(boxes.shape)
                        )
                else:
                    raise ValueError(
                        "Expected target boxes to be of type "
                        "Tensor, got {:}.".format(type(boxes))
                    )

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        self.valid_bbox(targets)

        img_features = self.backbone(images.tensors)

        if isinstance(img_features, torch.Tensor):
            img_features = OrderedDict([("0", img_features)])

        if self.setup.use_fixations:
            ### Deal with fixations ###
            fixations = self.transform(fixations)[0]
            fixations_features = self.backbone(fixations.tensors)
            if isinstance(fixations_features, torch.Tensor):
                fixations_features = OrderedDict([("0", fixations_features)])

            features = self.fuse_features(img_features, fixations_features)
        else:
            features = img_features

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features,
            proposals,
            images.image_sizes,
            targets,
        )
        detections = postprocess(
            self.transform, detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)


class MultimodalFasterRCNN(MultimodalGeneralizedRCNN):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import FasterRCNN
        >>> from torchvision.models.detection.rpn import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # FasterRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> # put the pieces together inside a FasterRCNN model
        >>> model = FasterRCNN(backbone,
        >>>                    num_classes=2,
        >>>                    rpn_anchor_generator=anchor_generator,
        >>>                    box_roi_pool=roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(
        self,
        setup: ModelSetup,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        fixation_backbone=None,
    ):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    "num_classes should be None when box_predictor is specified"
                )
        else:
            if box_predictor is None:
                raise ValueError(
                    "num_classes should not be None when box_predictor "
                    "is not specified"
                )

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test
        )

        rpn = XAMIRegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            box_head = XAMITwoMLPHead(
                setup,
                out_channels * resolution ** 2,
                setup.representation_size,
                dropout_rate=setup.box_head_dropout_rate,
            )

        if box_predictor is None:
            box_predictor = FastRCNNPredictor(setup.representation_size, num_classes)

        roi_heads = XAMIRoIHeads(
            # Box
            setup,
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(
            min_size,
            max_size,
            image_mean,
            image_std,
            fixed_size=[setup.image_size, setup.image_size],
        )

        super(MultimodalFasterRCNN, self).__init__(
            setup,
            backbone,
            rpn,
            roi_heads,
            transform,
            fixation_backbone=fixation_backbone,
        )


class MultimodalMaskRCNN(MultimodalFasterRCNN):
    """
    Implements Mask R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
        - masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        mask_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
             the locations indicated by the bounding boxes, which will be used for the mask head.
        mask_head (nn.Module): module that takes the cropped feature maps as input
        mask_predictor (nn.Module): module that takes the output of the mask_head and returns the
            segmentation mask logits

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import MaskRCNN
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>>
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # MaskRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                      output_size=14,
        >>>                                                      sampling_ratio=2)
        >>> # put the pieces together inside a MaskRCNN model
        >>> model = MaskRCNN(backbone,
        >>>                  num_classes=2,
        >>>                  rpn_anchor_generator=anchor_generator,
        >>>                  box_roi_pool=roi_pooler,
        >>>                  mask_roi_pool=mask_roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    def __init__(
        self,
        setup: ModelSetup,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # Mask parameters
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        fixation_backbone=None,
    ):

        assert isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError(
                    "num_classes should be None when mask_predictor is specified"
                )

        super(MultimodalMaskRCNN, self).__init__(
            setup,
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            fixation_backbone=fixation_backbone,
        )

        if setup.use_mask:

            out_channels = backbone.out_channels

            if mask_roi_pool is None:
                mask_roi_pool = MultiScaleRoIAlign(
                    featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
                )

            if mask_head is None:
                mask_layers = (256, 256, 256, 256)
                mask_dilation = 1
                mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

            if mask_predictor is None:
                mask_predictor_in_channels = 256  # == mask_layers[-1]
                mask_dim_reduced = 256
                mask_predictor = MaskRCNNPredictor(
                    mask_predictor_in_channels, mask_dim_reduced, num_classes
                )

            self.roi_heads.mask_roi_pool = mask_roi_pool
            self.roi_heads.mask_head = mask_head
            self.roi_heads.mask_predictor = mask_predictor

def postprocess(
    transform,
    result: List[Dict[str, Tensor]],
    image_shapes: List[Tuple[int, int]],
    original_image_sizes: List[Tuple[int, int]],
) -> List[Dict[str, Tensor]]:
    # if self.training:
    #     return result
    for i, (pred, im_s, o_im_s) in enumerate(
        zip(result, image_shapes, original_image_sizes)
    ):
        boxes = pred["boxes"]
        boxes = resize_boxes(boxes, im_s, o_im_s)
        result[i]["boxes"] = boxes
        if "masks" in pred:
            masks = pred["masks"]
            masks = paste_masks_in_image(masks, boxes, o_im_s)
            result[i]["masks"] = masks
        if "keypoints" in pred:
            keypoints = pred["keypoints"]
            keypoints = resize_keypoints(keypoints, im_s, o_im_s)
            result[i]["keypoints"] = keypoints
    return result
