from shutil import ExecError
from time import time
from tkinter import EXCEPTION
from setuptools import setup
import torch
import warnings
import torchvision
import itertools
import math
import torchvision.models.detection._utils as det_utils
from collections import OrderedDict
from torch import nn, Tensor
from typing import Tuple, List, Dict, Union, Optional
import torch.nn.functional as F

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.faster_rcnn import (
    MultiScaleRoIAlign,
    RPNHead,
    # RegionProposalNetwork,
    RoIHeads,
    # TwoMLPHead,
    FastRCNNPredictor,
    AnchorGenerator,
    GeneralizedRCNNTransform,
)

from torchvision.models.detection.roi_heads import (
    fastrcnn_loss,
    maskrcnn_inference,
    maskrcnn_loss,
)

from torchvision.models.detection.rpn import (
    _onnx_get_num_anchors_and_pre_nms_top_n,
    concat_box_prediction_layers,
)


from torchvision.models.detection.image_list import ImageList

from torchvision.models.detection.transform import resize_boxes, resize_keypoints
from torchvision.models.detection.roi_heads import paste_masks_in_image

from models.setup import ModelSetup


class XAMITwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(
        self, setup: ModelSetup, in_channels, representation_size, dropout_rate=0
    ):
        super().__init__()

        self.setup = setup
        self.fc6 = nn.Sequential(
            nn.Linear(
                in_channels + setup.clinical_input_channels
                if setup.add_clinical_to_roi_heads
                else in_channels,
                representation_size,
            ),
            nn.Dropout2d(p=dropout_rate, inplace=False),
        )
        self.fc7 = nn.Sequential(
            nn.Linear(representation_size, representation_size),
            nn.Dropout2d(p=dropout_rate, inplace=False),
        )

    def forward(self, x, clinical_input=None):
        self.x = x
        x = x.flatten(start_dim=1)

        if self.setup.add_clinical_to_roi_heads:
            assert (
                not clinical_input is None
            ), "You design the model to attach the clinical data in the box_pred; however, no clincal input is being provided."
            self.clinical_input = clinical_input
            x = torch.concat([x, clinical_input], axis=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class XAMIRegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(
                    anchors_per_image.shape, dtype=torch.float32, device=device
                )
                labels_per_image = torch.zeros(
                    (anchors_per_image.shape[0],), dtype=torch.float32, device=device
                )
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = (
                    matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                )
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(
        self, objectness: Tensor, num_anchors_per_level: List[int]
    ) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(
                    ob, self.pre_nms_top_n()
                )
            else:
                num_anchors = ob.shape[1]
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(
            proposals, objectness_prob, levels, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(
        self,
        objectness: Tensor,
        pred_bbox_deltas: Tensor,
        labels: List[Tensor],
        regression_targets: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [
            s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
        ]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(
            proposals, objectness, images.image_sizes, num_anchors_per_level
        )

        losses = {}
        if not targets is None:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses


class XAMIRoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        setup: ModelSetup,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        use_gt_in_train=True,
        across_class_nms_thresh=None,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        self.setup = setup
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.use_gt_in_train = use_gt_in_train

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.across_class_nms_thresh= across_class_nms_thresh

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
            proposals, gt_boxes, gt_labels
        ):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(
                    gt_boxes_in_image, proposals_in_image
                )
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = (
                    matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                )
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = (
                    matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                )
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])
        if self.has_mask():
            assert all(["masks" in t for t in targets])

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos (facilitate the training.)
        if self.training and self.use_gt_in_train:
            proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels
        )
        # sample a f proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            if not self.across_class_nms_thresh:
                keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if self.across_class_nms_thresh:
                keep = box_ops.batched_nms(boxes, scores, torch.ones_like(labels), self.across_class_nms_thresh)
                keep = keep[: self.detections_per_img]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
        clinical_input=None,
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert (
                    t["boxes"].dtype in floating_point_types
                ), "target boxes must of float type"
                assert (
                    t["labels"].dtype == torch.int64
                ), "target labels must of int64 type"
                if self.has_keypoint():
                    assert (
                        t["keypoints"].dtype == torch.float32
                    ), "target keypoints must of float type"

        if targets:
            (
                proposals,
                matched_idxs,
                labels,
                regression_targets,
            ) = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        self.proposals = proposals

        if self.setup.add_clinical_to_roi_heads:
            # you will expect to receive the clinical_input
            assert (
                not clinical_input is None
            ), "You design the model to attach the clinical data in the roi_head; however, no clincal input is being provided."
            proposals_clinical_input = torch.concat(
                [
                    clinical_input[i].repeat(p.shape[0], 1)
                    for i, p in enumerate(proposals)
                ],
                axis=0,
            )
            box_features = self.box_head(box_features, proposals_clinical_input)
        else:
            box_features = self.box_head(box_features)

        class_logits, box_regression = self.box_predictor(box_features)
        self.pred_out_logits, self.pred_out_reg = class_logits, box_regression

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}

        if targets:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        pred_boxes, pred_scores, pred_labels = self.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )

        num_images = len(pred_boxes)

        for i in range(num_images):
            result.append(
                {
                    "boxes": pred_boxes[i],
                    "labels": pred_labels[i],
                    "scores": pred_scores[i],
                }
            )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            eval_mask_proposals = [p["boxes"] for p in result]

            if targets:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(
                    features, mask_proposals, image_shapes
                )
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)

                eval_mask_features = self.mask_roi_pool(
                    features, eval_mask_proposals, image_shapes
                )
                eval_mask_features = self.mask_head(eval_mask_features)
                eval_mask_logits = self.mask_predictor(eval_mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if targets:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
                loss_mask = {"loss_mask": rcnn_loss_mask}

            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(eval_mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

            losses.update(loss_mask)

        return result, losses


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
        clinical_backbone=None,
    ):

        super(MultimodalGeneralizedRCNN, self).__init__()

        self.transform = transform
        self.backbone = backbone
        self.fusing_channels = self.backbone.out_channels
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

        self.clinical_convs = clinical_backbone
        self.setup = setup

        example_img_features = self.backbone(
            self.transform([torch.ones(3, 2048, 2048)])[0].tensors
        )

        # if isinstance(example_img_features, OrderedDict):
        if isinstance(example_img_features, torch.Tensor):
            self.feature_keys = ["0"]
            example_img_features = OrderedDict([("0", example_img_features)])
        else:
            self.feature_keys = example_img_features.keys()

        last_key = list(example_img_features.keys())[-1]
        self.image_feature_map_size = example_img_features[last_key].shape[-1]

        if self.setup.use_clinical:
            self.build_clinical_model()

    def build_clinical_model(self,):
        self.gender_emb_layer = nn.Embedding(
            2, self.setup.clinical_input_channels - self.setup.clinical_num_len
        )

        if self.setup.spatialise_clinical:
            if self.setup.spatialise_method == "convs":
                expand_times = math.log((self.setup.image_size), 2)

                assert (
                    expand_times.is_integer(),
                    f"The expand_times should be interger but found {expand_times}",
                )

                expand_conv_modules = list(
                    itertools.chain.from_iterable(
                        [
                            [
                                nn.ConvTranspose2d(
                                    (
                                        self.setup.clinical_input_channels
                                        if i == 0
                                        else self.setup.clinical_expand_conv_channels
                                    ),
                                    (self.setup.clinical_expand_conv_channels),
                                    kernel_size=2,
                                    stride=2,
                                ),
                                nn.BatchNorm2d(
                                    (self.setup.clinical_expand_conv_channels),
                                ),
                                nn.ReLU(inplace=False),
                                nn.Conv2d(
                                    (self.setup.clinical_expand_conv_channels),
                                    (
                                        3
                                        if i == int(expand_times) - 1
                                        else self.setup.clinical_expand_conv_channels
                                    ),
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                ),
                                nn.BatchNorm2d(
                                    (
                                        3
                                        if i == int(expand_times) - 1
                                        else self.setup.clinical_expand_conv_channels
                                    ),
                                ),
                            ]
                            + (
                                []
                                if i == expand_times - 1
                                else [nn.ReLU(inplace=False),]
                            )
                            for i in range(
                                int(expand_times)
                            )  # expand to the same as image size.
                        ]
                    )
                )

                self.clinical_expand_conv = nn.Sequential(*expand_conv_modules,)
            elif self.setup.spatialise_method == "repeat":
                self.before_repeat = nn.ModuleDict({})
                for k in self.feature_keys:
                    self.before_repeat[k] = nn.Linear(
                        self.setup.clinical_input_channels, self.fusing_channels
                    )
            else:
                raise Exception(
                    f"Unsupported spatialisation method: {self.setup.spatialise_method}"
                )

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
                        else self.setup.clinical_conv_channels
                    ),
                    (
                        self.fusing_channels
                        if i == (self.setup.fuse_depth - 1)
                        else self.setup.clinical_conv_channels
                    ),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(
                    (
                        self.fusing_channels
                        if i == (self.setup.fuse_depth - 1)
                        else self.setup.clinical_conv_channels
                    )
                ),
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
                                else self.setup.clinical_conv_channels
                            ),
                            (
                                self.fusing_channels
                                if i == (self.setup.fuse_depth - 1)
                                else self.setup.clinical_conv_channels
                            ),
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(
                            (
                                self.fusing_channels
                                if i == (self.setup.fuse_depth - 1)
                                else self.setup.clinical_conv_channels
                            )
                        ),
                        nn.ReLU(inplace=False),
                    ]
                    for i in range(self.setup.fuse_depth)
                ]
            )
        )

        self.fuse_convs = nn.Sequential(*fuse_convs_modules)

    def get_fuse_input_channel(self,):
        if self.setup.fusion_strategy == "concat":
            return self.fusing_channels * 2
        elif self.setup.fusion_strategy == "add":
            return self.fusing_channels
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

    def forward(self, images, clinical_num=None, clinical_cat=None, targets=None):

        """
        Args
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.setup.use_clinical:
            assert (not clinical_num is None) or (
                not clinical_cat is None
            ), "You're using clinical data, but they're not passed into model."

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
        # TODO: Move this to a function
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

        img_features = self.backbone(images.tensors)

        if isinstance(img_features, torch.Tensor):
            img_features = OrderedDict([("0", img_features)])

        clinical_input = None
        if self.setup.use_clinical:
            clinical_input, clinical_features = self.get_clinical_features(
                clinical_num, clinical_cat, img_features
            )

            if self.setup.spatialise_clinical:
                features = self.fuse_features(img_features, clinical_features)
            else:
                features = img_features
        else:
            features = img_features

        proposals, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses = self.roi_heads(
            features,
            proposals,
            images.image_sizes,
            targets,
            clinical_input=clinical_input,
        )

        detections = postprocess(
            self.transform, detections, images.image_sizes, original_image_sizes
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "RCNN always returns a (Losses, Detections) tuple in scripting"
                )
                self._has_warned = True
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
        clinical_backbone=None,
        across_class_nms_thresh=None,
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
            across_class_nms_thresh= across_class_nms_thresh,
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
            clinical_backbone=clinical_backbone,
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
        clinical_backbone=None,
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
            clinical_backbone=clinical_backbone,
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


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = nn.Conv2d(
                next_feature,
                layer_features,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
            )
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=False)
            next_feature = layer_features

        super(MaskRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(MaskRCNNPredictor, self).__init__(
            OrderedDict(
                [
                    (
                        "conv5_mask",
                        nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0),
                    ),
                    ("relu", nn.ReLU(inplace=False)),
                    ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)


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
