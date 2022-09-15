from typing import Dict, List

XAMI_MIMIC_PATH = "D:\XAMI-MIMIC"
SPREADSHEET_FOLDER = "spreadsheets"

"""
MIMIC
"""
DEFAULT_MIMIC_CLINICAL_NUM_COLS: List[str] = [
    "age",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "acuity",
]



# DEFAULT_MIMIC_CLINICAL_NUM_COLS: List[str] = [
#     "age",
#     "temperature",
#     "heartrate",
#     "resprate",
#     "o2sat",
#     "sbp",
#     "dbp",
#     "pain",
#     "acuity",
# ]


DEFAULT_MIMIC_CLINICAL_CAT_COLS: List[str] = ["gender"]

"""
REFLACX
"""

DEFAULT_REFLACX_LABEL_COLS: List[str] = [
    "Enlarged cardiac silhouette",
    "Atelectasis",
    "Pleural abnormality",
    "Consolidation",
    "Pulmonary edema",
    #  'Groundglass opacity', #6th disease.
]

DEFAULT_REFLACX_ALL_DISEASES: List[str] = [
    "Airway wall thickening",
    "Atelectasis",
    "Consolidation",
    "Enlarged cardiac silhouette",
    "Fibrosis",
    "Groundglass opacity",
    "Pneumothorax",
    "Pulmonary edema",
    "Wide mediastinum",
    "Abnormal mediastinal contour",
    "Acute fracture",
    "Enlarged hilum",
    "Hiatal hernia",
    "High lung volume / emphysema",
    "Interstitial lung disease",
    "Lung nodule or mass",
    "Pleural abnormality",
]

DEFAULT_REFLACX_REPETITIVE_LABEL_MAP: Dict[str, List[str]] = {
    "Airway wall thickening": ["Airway wall thickening"],
    "Atelectasis": ["Atelectasis"],
    "Consolidation": ["Consolidation"],
    "Enlarged cardiac silhouette": ["Enlarged cardiac silhouette"],
    "Fibrosis": ["Fibrosis"],
    "Groundglass opacity": ["Groundglass opacity"],
    "Pneumothorax": ["Pneumothorax"],
    "Pulmonary edema": ["Pulmonary edema"],
    "Quality issue": ["Quality issue"],
    "Support devices": ["Support devices"],
    "Wide mediastinum": ["Wide mediastinum"],
    "Abnormal mediastinal contour": ["Abnormal mediastinal contour"],
    "Acute fracture": ["Acute fracture"],
    "Enlarged hilum": ["Enlarged hilum"],
    "Hiatal hernia": ["Hiatal hernia"],
    "High lung volume / emphysema": ["High lung volume / emphysema", "Emphysema",],
    "Interstitial lung disease": ["Interstitial lung disease"],
    "Lung nodule or mass": ["Lung nodule or mass", "Mass", "Nodule"],
    "Pleural abnormality": [
        "Pleural abnormality",
        "Pleural thickening",
        "Pleural effusion",
    ],
}

DEFAULT_REFLACX_BOX_COORD_COLS: List[str] = ["xmin", "ymin", "xmax", "ymax"]
DEFAULT_REFLACX_BOX_FIX_COLS: List[str] = DEFAULT_REFLACX_BOX_COORD_COLS + ["certainty"]
DEFAULT_REFLACX_PATH_COLS : List[str]= [
    "image_path",
    "anomaly_location_ellipses_path",
    "bbox_paths",
    "fixations_path"
]

