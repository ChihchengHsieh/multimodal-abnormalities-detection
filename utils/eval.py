import pickle, os
from collections import OrderedDict
from typing import Tuple
from .coco_eval import CocoEvaluator, external_summarize

def get_ar_ap(
    evaluator: CocoEvaluator,
    areaRng: str = "all",
    iouThr: float = 0.5,
    maxDets: int = 10,
) -> Tuple[float, float]:

    ar = external_summarize(
        evaluator.coco_eval["bbox"],
        ap=0,
        areaRng=areaRng,
        iouThr=iouThr,
        maxDets=maxDets,
        print_result=False,
    )

    ap = external_summarize(
        evaluator.coco_eval["bbox"],
        ap=1,
        areaRng=areaRng,
        iouThr=iouThr,
        maxDets=maxDets,
        print_result=False,
    )

    return ar, ap


def save_iou_results(evaluator: CocoEvaluator, suffix: str, model_path: str):
    ap_ar_dict = OrderedDict(
        {thrs: [] for thrs in evaluator.coco_eval["bbox"].params.iouThrs}
    )

    for thrs in evaluator.coco_eval["bbox"].params.iouThrs:
        test_ar, test_ap = get_ar_ap(evaluator, areaRng="all", maxDets=10, iouThr=thrs,)

        ap_ar_dict[thrs].append(
            {"ar": test_ar, "ap": test_ap,}
        )

        print(f"IoBB [{thrs:.4f}] | AR [{test_ar:.4f}] | AP [{test_ap:.4f}]")

    with open(
        os.path.join("eval_results", f"{model_path}_{suffix}.pkl"), "wb",
    ) as training_record_f:
        pickle.dump(ap_ar_dict, training_record_f)

