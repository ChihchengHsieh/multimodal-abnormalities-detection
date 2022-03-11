import enum
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import utils.transforms as T
import os

from sklearn.preprocessing import LabelEncoder
from PIL import Image
import PIL

# import torchvision.transforms as torch_transform

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from torch.autograd import Variable
from utils.map import map_target_to_device


def collate_fn(batch):
        return tuple(zip(*batch))


class ReflacxDataset(data.Dataset):
    def __init__(
        self,
        XAMI_MIMIC_PATH,
        using_full_reflacx=True,
        with_clinical=False,
        bbox_to_mask=False,
        split_str=None,
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
        box_coord_cols=["xmin", "ymin", "xmax", "ymax"],
        path_cols=["image_path", "anomaly_location_ellipses_path"],
    ):
        # Data loading selections
        self.with_clinical = with_clinical
        self.using_full_reflacx = using_full_reflacx
        self.split_str = split_str

        # Image related
        self.image_size = image_size
        self.transforms = transforms

        # Labels
        self.labels_cols = labels_cols
        self.all_disease_cols = all_disease_cols
        self.repetitive_label_map = repetitive_label_map
        self.box_fix_cols = box_fix_cols
        self.box_coord_cols = box_coord_cols
        self.bbox_to_mask = bbox_to_mask

        if self.using_full_reflacx:
            assert (
                self.with_clinical == False
            ), "The full REFLACX dataset doesn't come with identified stayId; hence, it can't be used with clincal data."
            self.df = pd.read_csv("reflacx_cxr.csv", index_col=0)

        else:
            self.df = pd.read_csv("reflacx_with_clinical.csv", index_col=0)

        # determine if using clinical data.
        if self.with_clinical:
            ## initialise clinical fields.
            self.clinical_numerical_cols = clinical_numerical_cols
            self.clinical_categorical_cols = clinical_categorical_cols
            self.clinical_cols = clinical_numerical_cols + clinical_categorical_cols
            self.preprocess_clinical_df()

        ## Split dataset.
        if not self.split_str is None:
            self.df = self.df[self.df["split"] == self.split_str]

        ## repalce the correct path for mimic folder.
        for p_col in path_cols:
            self.df[p_col] = self.df[p_col].apply(
                lambda x: x.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH)
            )

        ## preprocessing data.
        self.preprocess_label()

        super(ReflacxDataset, self).__init__()

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

    def label_idx_to_disease(self, idx):
        if idx == 0:
            return "background"

        if idx > len(self.labels_cols):
            return f"exceed label range :{idx}"

        return self.labels_cols[idx - 1]

    def __len__(self):
        return len(self.df)

    def generate_bboxes_df(
        self, ellipse_df,
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
        boxes_df["label"] = boxes_df[self.labels_cols].idxmax(axis=1)
        boxes_df = boxes_df[self.box_fix_cols + ["label"]]

        return boxes_df


    def __getitem__(self, idx):
        # find the df
        data = self.df.iloc[idx]

        img = PIL.Image.open(data["image_path"]).convert("RGB")

        ## Get bounding boxes.
        bboxes_df = self.generate_bboxes_df(
            pd.read_csv(data["anomaly_location_ellipses_path"])
        )
        bboxes = torch.tensor(np.array(bboxes_df[self.box_coord_cols], dtype=float))

  
        ## Calculate area of boxes.
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        labels = torch.tensor(
            np.array(bboxes_df["label"].apply(lambda l: self.disease_to_idx(l))),
            dtype=torch.int64,
        )

        image_id = torch.tensor([idx])
        num_objs = bboxes.shape[0]

        ## suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # prepare all targets
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["dicom_id"] = data["dicom_id"]
        target["image_path"] = data["image_path"]


        if self.bbox_to_mask:
            # generate masks from bboxes
            masks = torch.zeros((num_objs, img.height, img.width), dtype=torch.uint8)
            for i, b in enumerate(bboxes):
                b = b.int()
                masks[i, b[1] : b[3], b[0] : b[2]] = 1
            target["masks"] = masks

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if self.with_clinical:

            clinical_num = torch.tensor(
                np.array(data[self.clinical_numerical_cols], dtype=float)
            ).float()

            clinical_cat = torch.tensor(
                np.array(data[self.clinical_categorical_cols], dtype=int)
            )

            return img, clinical_num, clinical_cat, target

        return img, target

    def prepare_input_from_data(self, data, device):

        if self.with_clinical:
            imgs, clinical_num, clinical_cat, targets = data

            imgs = list(img.to(device) for img in imgs)
            clinical_num = [t.to(device) for t in clinical_num]
            clinical_cat = [t.to(device) for t in clinical_cat]
            targets = [map_target_to_device(t, device) for t in targets]

            return (imgs, clinical_num, clinical_cat, targets)

        else:
            imgs, targets = data

            imgs = list(img.to(device) for img in imgs)
            targets = [map_target_to_device(t, device) for t in targets]

            return (imgs, targets)


class REFLACXWithClinicalAndBoundingBoxDataset(data.Dataset):
    def __init__(
        self,
        XAMI_MIMIC_PATH,
        split_str=None,
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
        box_coord_cols=["xmin", "ymin", "xmax", "ymax"],
        path_cols=["image_path", "anomaly_location_ellipses_path"],
    ):

        ## assign data
        self.split_str = split_str
        self.image_size = image_size
        self.clinical_numerical_cols = clinical_numerical_cols
        self.clinical_categorical_cols = clinical_categorical_cols
        self.clinical_cols = clinical_numerical_cols + clinical_categorical_cols
        self.labels_cols = labels_cols
        self.all_disease_cols = all_disease_cols
        self.repetitive_label_map = repetitive_label_map
        self.box_fix_cols = box_fix_cols
        self.box_coord_cols = box_coord_cols
        self.transforms = transforms

        # load dataframe.
        self.df = pd.read_csv("reflacx_with_clinical.csv", index_col=0)

        if not self.split_str is None:
            self.df = self.df[self.df["split"] == self.split_str]

        ## repalce the correct path for mimic folder.
        for p_col in path_cols:
            self.df[p_col] = self.df[p_col].apply(
                lambda x: x.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH)
            )

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

    def label_index_to_disease(self, idx):
        if idx == 0:
            return "background"

        if idx > len(self.labels_cols):
            return "exceed label range"

        return self.labels_cols[idx - 1]

    def __len__(self):
        return len(self.df)

    def generate_bboxes_df(
        self, ellipse_df,
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
        boxes_df["label"] = boxes_df[self.labels_cols].idxmax(axis=1)
        boxes_df = boxes_df[self.box_fix_cols + ["label"]]

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
        clinical_categorical_input = torch.tensor(
            np.array(data[self.clinical_categorical_cols], dtype=int)
        )

        ## Get bounding boxes.
        bboxes_df = self.generate_bboxes_df(
            pd.read_csv(data["anomaly_location_ellipses_path"])
        )
        bboxes = torch.tensor(np.array(bboxes_df[self.box_coord_cols], dtype=float))

        ## Calculate area of boxes.
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        labels = torch.tensor(
            np.array(bboxes_df["label"].apply(lambda l: self.disease_to_idx(l))),
            dtype=torch.int64,
        )
        image_id = torch.tensor([idx])
        num_objs = bboxes.shape[0]

        ## suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # prepare all targets
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["dicom_id"] = data["dicom_id"]
        target["image_path"] = data["image_path"]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, clinical_numerical_input, clinical_categorical_input, target


class REFLACXWithBoundingBoxesDataset(data.Dataset):
    def __init__(
        self,
        XAMI_MIMIC_PATH,
        split_str=None,
        transforms=None,
        image_size=224,
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
        box_coord_cols=["xmin", "ymin", "xmax", "ymax"],
        path_cols=["image_path", "anomaly_location_ellipses_path"],
    ):

        ## assign data
        self.split_str = split_str
        self.image_size = image_size
        self.labels_cols = labels_cols
        self.all_disease_cols = all_disease_cols
        self.repetitive_label_map = repetitive_label_map
        self.box_fix_cols = box_fix_cols
        self.box_coord_cols = box_coord_cols
        self.transforms = transforms

        # load dataframe.
        self.df = pd.read_csv("reflacx_with_clinical.csv", index_col=0)

        if not self.split_str is None:
            self.df = self.df[self.df["split"] == self.split_str]

        ## repalce the correct path for mimic folder.
        for p_col in path_cols:
            self.df[p_col] = self.df[p_col].apply(
                lambda x: x.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH)
            )

        ## preprocessing data.
        self.preprocess_label()

        super(REFLACXWithBoundingBoxesDataset, self).__init__()

    def preprocess_label(self,):
        self.df[self.all_disease_cols] = self.df[self.all_disease_cols].gt(0)

    def load_image_array(self, image_path):
        return np.asarray(Image.open(image_path))

    def plot_image_from_array(self, image_array):
        im = Image.fromarray(image_array)
        im.show()

    def plot_pil_image(self, image):
        image.show()

    def disease_to_idx(self, disease):
        if not disease in self.labels_cols:
            raise Exception("This disease is not the label.")

        return self.labels_cols.index(disease) + 1

    def label_index_to_disease(self, idx):
        if idx == 0:
            return "background"

        if idx > len(self.labels_cols):
            return "exceed label range"

        return self.labels_cols[idx - 1]

    def __len__(self):
        return len(self.df)

    def generate_boxes_df(
        self, ellipse_df,
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
        boxes_df["label"] = boxes_df[self.labels_cols].idxmax(axis=1)
        boxes_df = boxes_df[self.box_fix_cols + ["label"]]

        return boxes_df

    def collate_fn(batch):
        return tuple(zip(*batch))

    def __getitem__(self, idx):
        # find the df
        data = self.df.iloc[idx]

        img = Image.open(data["image_path"]).convert("RGB")

        ## Get bounding boxes.
        boxes_df = self.generate_boxes_df(
            pd.read_csv(data["anomaly_location_ellipses_path"])
        )
        boxes = torch.tensor(np.array(boxes_df[self.box_coord_cols], dtype=float))

        ## Calculate area of boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        labels = torch.tensor(
            np.array(boxes_df["label"].apply(lambda l: self.disease_to_idx(l))),
            dtype=torch.int64,
        )
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
        target["dicom_id"] = data["dicom_id"]
        target["image_path"] = data["image_path"]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class ReflacxAllCXRDataset(data.Dataset):
    def __init__(
        self,
        XAMI_MIMIC_PATH,
        split_str=None,
        transforms=None,
        image_size=224,
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
        box_coord_cols=["xmin", "ymin", "xmax", "ymax"],
        path_cols=["image_path", "anomaly_location_ellipses_path"],
    ):

        ## assign data
        self.split_str = split_str
        self.image_size = image_size
        self.labels_cols = labels_cols
        self.all_disease_cols = all_disease_cols
        self.repetitive_label_map = repetitive_label_map
        self.box_fix_cols = box_fix_cols
        self.box_coord_cols = box_coord_cols
        self.transforms = transforms

        # load dataframe.
        self.df = pd.read_csv("reflacx_cxr.csv", index_col=0)

        if not self.split_str is None:
            self.df = self.df[self.df["split"] == self.split_str]

        ## repalce the correct path for mimic folder.
        for p_col in path_cols:
            self.df[p_col] = self.df[p_col].apply(
                lambda x: x.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH)
            )

        ## preprocessing data.
        self.preprocess_label()

        super(ReflacxAllCXRDataset, self).__init__()

    def preprocess_label(self,):
        self.df[self.all_disease_cols] = self.df[self.all_disease_cols].gt(0)

    def load_image_array(self, image_path):
        return np.asarray(Image.open(image_path))

    def plot_image_from_array(self, image_array):
        im = Image.fromarray(image_array)
        im.show()

    def plot_pil_image(self, image):
        image.show()

    def disease_to_idx(self, disease):
        if not disease in self.labels_cols:
            raise Exception("This disease is not the label.")

        return self.labels_cols.index(disease) + 1

    def label_index_to_disease(self, idx):
        if idx == 0:
            return "background"

        if idx > len(self.labels_cols):
            return "exceed label range"

        return self.labels_cols[idx - 1]

    def __len__(self):
        return len(self.df)

    def generate_boxes_df(
        self, ellipse_df,
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
        boxes_df["label"] = boxes_df[self.labels_cols].idxmax(axis=1)
        boxes_df = boxes_df[self.box_fix_cols + ["label"]]

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

        clinical_categorical_input = torch.tensor(
            np.array(data[self.clinical_categorical_cols], dtype=int)
        )

        ## Get bounding boxes.
        boxes_df = self.generate_boxes_df(
            pd.read_csv(data["anomaly_location_ellipses_path"])
        )
        boxes = torch.tensor(np.array(boxes_df[self.box_coord_cols], dtype=float))

        ## Calculate area of boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        labels = torch.tensor(
            np.array(boxes_df["label"].apply(lambda l: self.disease_to_idx(l))),
            dtype=torch.int64,
        )
        image_id = torch.tensor([idx])
        num_objs = boxes.shape[0]

        ## suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # generate masks from bboxes
        masks = torch.zeros((num_objs, img.height, img.width), dtype=torch.uint8)
        for i, b in enumerate(boxes):
            b = b.int()
            masks[i, b[1] : b[3], b[0] : b[2]] = 1

        # prepare all targets
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["dicom_id"] = data["dicom_id"]
        target["image_path"] = data["image_path"]
        target["masks"] = masks

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_path"] = img_path

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
