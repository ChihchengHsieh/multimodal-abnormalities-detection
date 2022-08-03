import pickle, os
from collections import OrderedDict
from typing import Tuple
from .coco_eval import CocoEvaluator, external_summarize, external_get_num_fps, external_get_num_fns, external_get_num_tps

# def get_ar_ap(
#     evaluator: CocoEvaluator,
#     areaRng: str = "all",
#     iouThr: float = 0.5,
#     maxDets: int = 10,
# ) -> Tuple[float, float]:

#     ar = external_summarize(
#         evaluator.coco_eval["bbox"],
#         ap=0,
#         areaRng=areaRng,
#         iouThr=iouThr,
#         maxDets=maxDets,
#         print_result=False,
#     )

#     ap = external_summarize(
#         evaluator.coco_eval["bbox"],
#         ap=1,
#         areaRng=areaRng,
#         iouThr=iouThr,
#         maxDets=maxDets,
#         print_result=False,
#     )

#     return ar, ap

def get_ap_ar(
    evaluator, iouThr=0.5, areaRng="all", maxDets=10,
):
    ap = external_summarize(
        evaluator.coco_eval["bbox"],
        ap=1,
        iouThr=iouThr,
        areaRng=areaRng,
        maxDets=maxDets,
        print_result=False,
    )

    ar = external_summarize(
        evaluator.coco_eval["bbox"],
        ap=0,
        iouThr=iouThr,
        areaRng=areaRng,
        maxDets=maxDets,
        print_result=False,
    )

    return {"ap": ap, "ar": ar}

def get_num_fps(
    evaluator, iouThr=0.5, areaRng="all", maxDets=10,
):
    num_fps = external_get_num_fps(
        evaluator.coco_eval["bbox"],
        iouThr=iouThr,
        areaRng=areaRng,
        maxDets=maxDets,
    )

    return num_fps

def get_num_fns(
    evaluator, iouThr=0.5, areaRng="all", maxDets=10,
):
    num_fns = external_get_num_fns(
        evaluator.coco_eval["bbox"],
        iouThr=iouThr,
        areaRng=areaRng,
        maxDets=maxDets,
    )

    return num_fns

def get_num_tps(
    evaluator, iouThr=0.5, areaRng="all", maxDets=10,
):
    num_tps = external_get_num_tps(
        evaluator.coco_eval["bbox"],
        iouThr=iouThr,
        areaRng=areaRng,
        maxDets=maxDets,
    )

    return num_tps

def get_ap_ar_for_train_val(
    train_evaluator: CocoEvaluator,
    val_evaluator: CocoEvaluator,
    iouThr=0.5,
    areaRng="all",
    maxDets=10,
):

    train_ap_ar = get_ap_ar(
        train_evaluator, iouThr=iouThr, areaRng=areaRng, maxDets=maxDets,
    )

    val_ap_ar = get_ap_ar(
        val_evaluator, iouThr=iouThr, areaRng=areaRng, maxDets=maxDets,
    )

    return train_ap_ar, val_ap_ar

def save_iou_results(evaluator: CocoEvaluator, suffix: str, model_path: str):
    ap_ar_dict = OrderedDict(
        {thrs: [] for thrs in evaluator.coco_eval["bbox"].params.iouThrs}
    )

    for thrs in evaluator.coco_eval["bbox"].params.iouThrs:
        test_ap_ar = get_ap_ar(evaluator, areaRng="all", maxDets=10, iouThr=thrs,)

        ap_ar_dict[thrs].append(
            test_ap_ar
        )

        print(f"IoBB [{thrs:.4f}] | AR [{test_ap_ar['ar']:.4f}] | AP [{test_ap_ar['ap']:.4f}]")

    with open(
        os.path.join("eval_results", f"{model_path}_{suffix}.pkl"), "wb",
    ) as training_record_f:
        pickle.dump(ap_ar_dict, training_record_f)

