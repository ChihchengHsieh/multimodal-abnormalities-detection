import json, copy, torch, torch._six
import numpy as np
import pycocotools.mask as mask_util


from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from collections import defaultdict
from . import detect_utils
from .coco_utils import get_coco_api_from_dataset
from pycocotools.cocoeval import Params


def get_eval_params_dict(
    dataset, iou_thrs=None, max_dets=None, thrs_start_at=0.5, use_iobb=True,
):

    iou_thrs = (
        iou_thrs
        if not iou_thrs is None
        else np.linspace(
            thrs_start_at,
            0.95,
            int(np.round((0.95 - thrs_start_at) / 0.05)) + 1,
            endpoint=True,
        )
    )

    max_dets = max_dets if not max_dets is None else [1, 10]

    eval_params_dict = {
        iou_type: Params(iouType=iou_type) for iou_type in ["bbox", "segm"]
    }

    eval_params_dict["bbox"].iouThrs = iou_thrs
    eval_params_dict["segm"].iouThrs = iou_thrs

    eval_params_dict["bbox"].maxDets = [1, 10]
    eval_params_dict["segm"].maxDets = [1, 10]

    coco_gt = get_coco_api_from_dataset(dataset)

    if not coco_gt is None:
        for k in eval_params_dict.keys():
            eval_params_dict[k].imgIds = sorted(coco_gt.getImgIds())
            eval_params_dict[k].catIds = sorted(coco_gt.getCatIds())

    eval_params_dict["bbox"].useIoBB = use_iobb
    eval_params_dict["segm"].useIoBB = False

    return eval_params_dict


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, params_dict=None):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

            if not (params_dict is None) and iou_type in params_dict.keys():
                self.coco_eval[iou_type].params = params_dict[iou_type]

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            summarize(coco_eval)
            # coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(
                    np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F")
                )[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = detect_utils.all_gather(img_ids)
    all_eval_imgs = detect_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions


def createIndex(self):
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if "annotations" in self.dataset:
        for ann in self.dataset["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
            anns[ann["id"]] = ann

    if "images" in self.dataset:
        for img in self.dataset["images"]:
            imgs[img["id"]] = img

    if "categories" in self.dataset:
        for cat in self.dataset["categories"]:
            cats[cat["id"]] = cat

    if "annotations" in self.dataset and "categories" in self.dataset:
        for ann in self.dataset["annotations"]:
            catToImgs[ann["category_id"]].append(ann["image_id"])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


maskUtils = mask_util


def external_get_num_fps(
    evaluator, iouThr=None, areaRng="all", maxDets=100,
):
    p = evaluator.params
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

    # dimension of precision: [TxRxKxAxM]
    s = evaluator.eval["num_fps"]
    # IoU
   # dimension of recall: [TxKxAxM]
    if iouThr is not None:
        if isinstance(iouThr, float):
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        elif isinstance(iouThr, list):
            t_start = np.where(iouThr[0] == p.iouThrs)[0]
            t_end = np.where(iouThr[1] == p.iouThrs)[0]
            s = s[t_start[0] : (t_end[0] + 1)]
    s = s[:, :, aind, mind]

    if len(s[s > -1]) == 0:
        sum_s = -1
    else:
        sum_s = np.sum(s[s > -1])
        
    return sum_s

def external_get_num_fns(
    evaluator, iouThr=None, areaRng="all", maxDets=100,
):
    p = evaluator.params
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

    # dimension of precision: [TxRxKxAxM]
    s = evaluator.eval["num_fns"]
    # IoU
   # dimension of recall: [TxKxAxM]
    if iouThr is not None:
        if isinstance(iouThr, float):
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        elif isinstance(iouThr, list):
            t_start = np.where(iouThr[0] == p.iouThrs)[0]
            t_end = np.where(iouThr[1] == p.iouThrs)[0]
            s = s[t_start[0] : (t_end[0] + 1)]
    s = s[:, :, aind, mind]

    if len(s[s > -1]) == 0:
        sum_s = -1
    else:
        sum_s = np.sum(s[s > -1])
        
    return sum_s

def external_get_num_tps(
    evaluator, iouThr=None, areaRng="all", maxDets=100,
):
    p = evaluator.params
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

    # dimension of precision: [TxRxKxAxM]
    s = evaluator.eval["num_tps"]
    # IoU
   # dimension of recall: [TxKxAxM]
    if iouThr is not None:
        if isinstance(iouThr, float):
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        elif isinstance(iouThr, list):
            t_start = np.where(iouThr[0] == p.iouThrs)[0]
            t_end = np.where(iouThr[1] == p.iouThrs)[0]
            s = s[t_start[0] : (t_end[0] + 1)]
    s = s[:, :, aind, mind]

    if len(s[s > -1]) == 0:
        sum_s = -1
    else:
        sum_s = np.sum(s[s > -1])
        
    return sum_s

def external_summarize(
    evaluator, ap=1, iouThr=None, areaRng="all", maxDets=100, print_result=False,
):
    p = evaluator.params
    io_type_str = "IoBB" if p.useIoBB == True else " IoU"
    titleStr = "Average Precision" if ap == 1 else "Average Recall"
    typeStr = "(AP)" if ap == 1 else "(AR)"

    if iouThr is not None:
        if isinstance(iouThr, float):
            f"{iouThr:0.2f}"
        elif isinstance(iouThr, list):
            iouStr = f"{iouThr[0]:0.2f}:{iouThr[-1]:0.2f}"
            
    else:
        iouStr = f"{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}"

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = evaluator.eval["precision"]
        # IoU
        if iouThr is not None:
            if isinstance(iouThr, float):
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            elif isinstance(iouThr, list):
                t_start = np.where(iouThr[0] == p.iouThrs)[0]
                t_end = np.where(iouThr[1] == p.iouThrs)[0]
                s = s[t_start[0] : (t_end[0] + 1)]

        s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = evaluator.eval["recall"]
        if iouThr is not None:
            if isinstance(iouThr, float):
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            elif isinstance(iouThr, list):
                t_start = np.where(iouThr[0] == p.iouThrs)[0]
                t_end = np.where(iouThr[1] == p.iouThrs)[0]
                s = s[t_start[0] : (t_end[0] + 1)]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])

    if print_result:
        print(f"{titleStr:<18} {typeStr} @[ {io_type_str}={iouStr:<9} | area={areaRng:>6s} | maxDets={maxDets:>3d} ] = {mean_s:0.3f}")

    return mean_s


def summarize(self):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
        p = self.params
        io_type_str = "IoBB" if p.useIoBB == True else " IoU"
        iStr = " {:<18} {} @[ {}={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
        titleStr = "Average Precision" if ap == 1 else "Average Recall"
        typeStr = "(AP)" if ap == 1 else "(AR)"
        iouStr = (
            "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
            if iouThr is None
            else "{:0.2f}".format(iouThr)
        )

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval["precision"]
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval["recall"]
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        print(
            iStr.format(
                titleStr, typeStr, io_type_str, iouStr, areaRng, maxDets, mean_s
            )
        )
        return mean_s

    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[1])
        stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[1])
        stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[1])
        stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[1])
        stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[1])
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[1])
        stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[1])
        stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[1])
        return stats

    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
        stats[3] = _summarize(1, maxDets=20, areaRng="medium")
        stats[4] = _summarize(1, maxDets=20, areaRng="large")
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
        stats[8] = _summarize(0, maxDets=20, areaRng="medium")
        stats[9] = _summarize(0, maxDets=20, areaRng="large")
        return stats

    if not self.eval:
        raise Exception("Please run accumulate() first")
    iouType = self.params.iouType
    if iouType == "segm" or iouType == "bbox":
        summarize = _summarizeDets
    elif iouType == "keypoints":
        summarize = _summarizeKps
    self.stats = summarize()


def loadRes(self, resFile):
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """
    res = COCO()
    res.dataset["images"] = [img for img in self.dataset["images"]]

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, torch._six.string_classes):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile
    assert type(anns) == list, "results in not an array of objects"
    annsImgIds = [ann["image_id"] for ann in anns]
    assert set(annsImgIds) == (
        set(annsImgIds) & set(self.getImgIds())
    ), "Results do not correspond to current coco set"
    if "caption" in anns[0]:
        imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
            [ann["image_id"] for ann in anns]
        )
        res.dataset["images"] = [
            img for img in res.dataset["images"] if img["id"] in imgIds
        ]
        for id, ann in enumerate(anns):
            ann["id"] = id + 1
    elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            bb = ann["bbox"]
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if "segmentation" not in ann:
                ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann["area"] = bb[2] * bb[3]
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "segmentation" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            # now only support compressed RLE format as segmentation results
            ann["area"] = maskUtils.area(ann["segmentation"])
            if "bbox" not in ann:
                ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "keypoints" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            s = ann["keypoints"]
            x = s[0::3]
            y = s[1::3]
            x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann["area"] = (x2 - x1) * (y2 - y1)
            ann["id"] = id + 1
            ann["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset["annotations"] = anns
    createIndex(res)
    return res


# this is the original compute IoU function, so we use this to replace the original one, and try if it works.


def get_compute_IoU_func(evaluator):
    def computeIoU_func(imgId, catId):
        return computeIoU(evaluator, imgId, catId)

    return computeIoU_func


def computeIoU(self, imgId, catId):
    p = self.params

    self.test_imgId = imgId
    self.test_catId = catId
    if p.useCats:
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
    else:
        gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
        dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
    if len(gt) == 0 and len(dt) == 0:
        return []
    inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
    dt = [dt[i] for i in inds]
    if len(dt) > p.maxDets[-1]:
        dt = dt[0 : p.maxDets[-1]]

    if p.iouType == "segm":
        g = [g["segmentation"] for g in gt]
        d = [d["segmentation"] for d in dt]
    elif p.iouType == "bbox":
        g = [g["bbox"] for g in gt]
        d = [d["bbox"] for d in dt]
    else:
        raise Exception("unknown iouType for iou computation")

    # compute iou between each dt and gt region
    iscrowd = [int(o["iscrowd"]) for o in gt]

    if p.iouType == "bbox" and p.useIoBB == True:
        ious = get_iobbs(d=d, g=g, iscrowd=iscrowd)
    else:
        ious = maskUtils.iou(d, g, iscrowd)

    # if (catId == 5):
    #     self.test_d = d
    #     self.test_g = g
    #     self.test_iscrowd = iscrowd
    #     self.test_ious = ious
    #     raise StopIteration
    return ious


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """

    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print(
            "useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType)
        )
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        # computeIoU = self.computeIoU # the function  in orignial cocoEval. # so we have to dig in that function and copy it out to make it customisable.
        computeIoU = get_compute_IoU_func(self)
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
def get_ious(d, g, iscrowd):
    ious = []
    for d_b in d:
        detection_ious = []
        for g_b in g:
            # print(d_b)
            # print(g_b)
            detection_ious.append(
                get_iou(get_input_coord_dict(d_b), get_input_coord_dict(g_b))
            )
        ious.append(detection_ious)
    return np.array(ious)


def get_iobbs(d, g, iscrowd):
    # print("Using IoBB")
    iobbs = []
    for d_b in d:
        d_iobbs = []
        for g_b in g:
            # print(d_b)
            # print(g_b)
            d_iobbs.append(
                get_iobb(
                    pred_bb=get_input_coord_dict(d_b), gt_bb=get_input_coord_dict(g_b),
                )
            )
        iobbs.append(d_iobbs)
    return np.array(iobbs)


def get_input_coord_dict(bb):
    return {"x1": bb[0], "x2": bb[0] + bb[2], "y1": bb[1], "y2": bb[1] + bb[3]}


def get_iobb(pred_bb, gt_bb):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    ## expect x, y, w, h

    # => form x1, x2, y1, y2

    assert pred_bb["x1"] < pred_bb["x2"]
    assert pred_bb["y1"] < pred_bb["y2"]
    assert gt_bb["x1"] < gt_bb["x2"]
    assert gt_bb["y1"] < gt_bb["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(pred_bb["x1"], gt_bb["x1"])
    y_top = max(pred_bb["y1"], gt_bb["y1"])
    x_right = min(pred_bb["x2"], gt_bb["x2"])
    y_bottom = min(pred_bb["y2"], gt_bb["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    pred_area = (pred_bb["x2"] - pred_bb["x1"]) * (pred_bb["y2"] - pred_bb["y1"])
    # gt_area = (gt_bb['x2'] - gt_bb['x1']) * (gt_bb['y2'] - gt_bb['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iobb = intersection_area / float(pred_area)
    assert iobb >= 0.0
    assert iobb <= 1.0
    return iobb


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    ## expect x, y, w, h

    # => form x1, x2, y1, y2

    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1["x1"], bb2["x1"])
    y_top = max(bb1["y1"], bb2["y1"])
    x_right = min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
