import pickle, os
from collections import OrderedDict
from utils.coco_eval import get_ar_ap

def save_iou_results(evaluator, suffix, model_path):
    ap_ar_dict = OrderedDict({thrs: []  for thrs in evaluator.coco_eval['bbox'].params.iouThrs})

    for thrs in evaluator.coco_eval['bbox'].params.iouThrs:
        test_ar, test_ap = get_ar_ap(
            evaluator,
            areaRng='all',
            maxDets= 10,
            iouThr=thrs,
        )

        ap_ar_dict[thrs].append({
            'ar': test_ar,
            'ap': test_ap,
        })

        print(f"IoU [{thrs:.4f}] | AR [{test_ar:.4f}] | AP [{test_ap:.4f}]")

    ## iouThr=0.3
    # 0.5699603174603174 0.23895678925202643 (custom, with clinical)
    # 0.5992460317460317 0.2928311315704941 (custom, without clinical)
    # 0.5244047619047619 0.22751787710826604 (original, without clinical)

    with open(
        os.path.join("eval_results", f"{model_path}_{suffix}.pkl"), "wb",
    ) as training_record_f:
        pickle.dump(ap_ar_dict, training_record_f)

### iouThr=0.3, With score thrs.
# 0.45178571428571423 0.20556006215252126 (custom, with clinical)
# 0.481547619047619 0.25642166456776067 (custom, without clinical)
# 0.36710317460317465 0.1927187078516919(original, without clinical)