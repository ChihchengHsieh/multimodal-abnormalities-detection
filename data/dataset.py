import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import utils.transforms as T

from sklearn.preprocessing import LabelEncoder
from PIL import Image

# import torchvision.transforms as torch_transform

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torch.autograd import Variable

class REFLACXWithClinicalAndBoundingBoxDataset(data.Dataset):
    def __init__(
        self,
        XAMI_MIMIC_PATH,
        split_str = None,
        transforms=None,
        image_size=224,
        clinical_numerical_cols=[
            "age",
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
            "pain",
            "acuity",
        ],
        clinical_categorical_cols=["gender"],
        labels_cols=[
            "Enlarged cardiac silhouette",
            "Atelectasis",
            "Pleural abnormality",
            "Consolidation",
            "Pulmonary edema",
            #  'Groundglass opacity', # 6th disease.
        ],
        all_disease_cols=[
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
        ],
        repetitive_label_map={
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
            "High lung volume / emphysema": [
                "High lung volume / emphysema",
                "Emphysema",
            ],
            "Interstitial lung disease": ["Interstitial lung disease"],
            "Lung nodule or mass": ["Lung nodule or mass", "Mass", "Nodule"],
            "Pleural abnormality": [
                "Pleural abnormality",
                "Pleural thickening",
                "Pleural effusion",
            ],
        },
        box_fix_cols=["xmin", "ymin", "xmax", "ymax", "certainty"],
        box_coord_cols = ["xmin", "ymin", "xmax", "ymax"],
        path_cols = ['image_path', 'anomaly_location_ellipses_path'],
    ):

        ## assign data
        self.split_str = split_str
        self.image_size = image_size
        self.clinical_numerical_cols = clinical_numerical_cols
        self.clinical_categorical_cols = clinical_categorical_cols
        self.clinical_cols = clinical_numerical_cols + clinical_categorical_cols 
        self.labels_cols = labels_cols
        self.all_disease_cols = all_disease_cols
        self.repetitive_label_map  = repetitive_label_map
        self.box_fix_cols = box_fix_cols
        self.box_coord_cols = box_coord_cols
        self.transforms = transforms

        # load dataframe.
        self.df = pd.read_csv("reflacx_with_clinical.csv", index_col=0)

        if not self.split_str is None:
            self.df  = self.df[self.df['split'] == self.split_str]
        
        ## repalce the correct path for mimic folder.
        for p_col in path_cols:
            self.df[p_col] = self.df[p_col].apply(lambda x: x.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH))

        ## preprocessing data.
        self.preprocess_clinical_df()
        self.preprocess_label()

        super(REFLACXWithClinicalAndBoundingBoxDataset, self).__init__()

    def preprocess_clinical_df(self,):
        self.encoders_map = {}

        # encode the categorical cols.
        for col in self.clinical_categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders_map[col] = le

    def preprocess_label(self,):
        self.df[self.all_disease_cols] = self.df[self.all_disease_cols].gt(0)

    def load_image_array(self, image_path):
        return np.asarray(Image.open(image_path))

    def plot_image_from_array(self, image_array):
        im = Image.fromarray(image_array)
        im.show()

    def disease_to_idx(self, disease):
        if not disease in self.labels_cols:
            raise Exception("This disease is not the label.") 

        return self.labels_cols.index(disease) + 1

    def __len__(self):
        return len(self.df)

    def generate_boxes_df(
        self, 
        ellipse_df,
    ):
        boxes_df = ellipse_df[self.box_fix_cols]

        ## relabel repetitive columns.
        for k in self.repetitive_label_map.keys():
            boxes_df[k] = ellipse_df[
                [l for l in self.repetitive_label_map[k] if l in ellipse_df.columns]
            ].any(axis=1)

        ## filtering out the diseases not in the label_cols
        boxes_df = boxes_df[boxes_df[self.labels_cols].any(axis=1)]

        ## get labels
        boxes_df['label'] = boxes_df[self.labels_cols].idxmax(axis= 1)
        boxes_df = boxes_df[self.box_fix_cols + ['label']]

        return boxes_df

    def collate_fn(batch):
        return tuple(zip(*batch))

    def __getitem__(self, idx):
        # find the df
        data = self.df.iloc[idx]

        img = Image.open(data["image_path"]).convert("RGB") 

        ## Prepare clinical data.
        clinical_numerical_input = torch.tensor(
            np.array(data[self.clinical_numerical_cols], dtype=float)
        ).float()
        clinical_categorical_input =  torch.tensor(np.array(data[self.clinical_categorical_cols], dtype=int))

        ## Get bounding boxes.
        boxes_df = self.generate_boxes_df(pd.read_csv(data['anomaly_location_ellipses_path']))
        boxes = torch.tensor(np.array(boxes_df[self.box_coord_cols], dtype=float))

        ## Calculate area of boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        labels = torch.tensor(np.array(boxes_df['label'].apply(lambda l: self.disease_to_idx(l))), dtype=torch.int64)
        image_id = torch.tensor([idx])
        num_objs = boxes.shape[0]

        ## suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # prepare all targets
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, clinical_numerical_input, clinical_categorical_input, target
