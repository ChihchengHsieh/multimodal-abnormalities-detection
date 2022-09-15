import os, torch, json

import pandas as pd
import numpy as np
import torch.utils.data as data
import pandas as pd
import numpy as np
import torch.utils.data as data

from sklearn.preprocessing import StandardScaler
from typing import Callable, Dict, List, Tuple, Union
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from copy import deepcopy

from models.setup import ModelSetup
from .constants import (
    DEFAULT_REFLACX_BOX_COORD_COLS,
    DEFAULT_REFLACX_BOX_FIX_COLS,
    DEFAULT_MIMIC_CLINICAL_CAT_COLS,
    DEFAULT_MIMIC_CLINICAL_NUM_COLS,
    DEFAULT_REFLACX_ALL_DISEASES,
    DEFAULT_REFLACX_LABEL_COLS,
    DEFAULT_REFLACX_PATH_COLS,
    DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
    SPREADSHEET_FOLDER,
)
from .helpers import map_target_to_device


def collate_fn(batch: Tuple) -> Tuple:
    return tuple(zip(*batch))
    

class ReflacxDataset(data.Dataset):
    """
    Class to load the preprocessed REFLACX master sheet. There `.csv` files are required to run this class.

    - `reflacx_cxr.csv`
    - `reflacx_with_clinical.csv`
    - `reflacx_u_df.csv`

    """

    def __init__(
        self,
        XAMI_MIMIC_PATH: str,
        with_clinical: bool = False,
        bbox_to_mask: bool = False,
        split_str: str = None,
        transforms: Callable[[Image.Image, Dict], Tuple[torch.Tensor, Dict]] = None,
        dataset_mode: str = "normal",
        clinical_numerical_cols: List[str] = DEFAULT_MIMIC_CLINICAL_NUM_COLS,
        clinical_categorical_cols: List[str] = DEFAULT_MIMIC_CLINICAL_CAT_COLS,
        labels_cols: List[str] = DEFAULT_REFLACX_LABEL_COLS,
        all_disease_cols: List[str] = DEFAULT_REFLACX_ALL_DISEASES,
        repetitive_label_map: Dict[
            str, List[str]
        ] = DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
        box_fix_cols: List[str] = DEFAULT_REFLACX_BOX_FIX_COLS,
        box_coord_cols: List[str] = DEFAULT_REFLACX_BOX_COORD_COLS,
        path_cols: List[str] = DEFAULT_REFLACX_PATH_COLS,
        normalise_clinical_num=False,
        spreadsheets_folder=SPREADSHEET_FOLDER,
    ):
        # Data loading selections
        self.with_clinical: bool = with_clinical
        self.split_str: str = split_str

        # Image related
        self.transforms: Callable[
            [Image.Image, Dict], Tuple[torch.Tensor, Dict]
        ] = transforms
        self.path_cols: List[str] = path_cols
        self.normalise_clinical_num = normalise_clinical_num

        # Labels
        self.labels_cols: List[str] = labels_cols
        self.all_disease_cols: List[str] = all_disease_cols
        self.repetitive_label_map: Dict[str, List[str]] = repetitive_label_map
        self.box_fix_cols: List[str] = box_fix_cols
        self.box_coord_cols: List[str] = box_coord_cols
        self.bbox_to_mask: bool = bbox_to_mask
        self.dataset_mode: str = dataset_mode

        if self.dataset_mode == "full":
            assert (
                self.with_clinical == False
            ), "The full REFLACX dataset doesn't come with identified stayId; hence, it can't be used with clincal data."
            self.df: pd.DataFrame = pd.read_csv(
                os.path.join(spreadsheets_folder, "reflacx_cxr.csv"), index_col=0
            )

        elif self.dataset_mode == "normal":
            self.df: pd.DataFrame = pd.read_csv(
                os.path.join(spreadsheets_folder, "reflacx_with_clinical.csv"),
                index_col=0,
            )
        elif self.dataset_mode == "unified":
            self.df: pd.DataFrame = pd.read_csv(
                os.path.join(spreadsheets_folder, "reflacx_u_df.csv"), index_col=0
            )

        # determine if using clinical data.
        if self.with_clinical:
            ## initialise clinical fields.
            self.clinical_numerical_cols: List[str] = clinical_numerical_cols
            self.clinical_categorical_cols: List[str] = clinical_categorical_cols
            self.clinical_cols: List[
                str
            ] = clinical_numerical_cols + clinical_categorical_cols
            self.clinical_num_norm: StandardScaler = StandardScaler().fit(
                self.df[self.clinical_numerical_cols]
            )

            self.preprocess_clinical_df()

        ## Split dataset.
        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        ## repalce the path with local mimic folder path.
        for p_col in path_cols:
            if p_col in self.df.columns:
                if p_col == "bbox_paths":

                    def apply_bbox_paths_transform(input_paths_str: str) -> List[str]:
                        input_paths_list: List[str] = json.loads(input_paths_str)
                        replaced_path_list: List[str] = [
                            p.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH)
                            for p in input_paths_list
                        ]
                        return replaced_path_list

                    apply_fn: Callable[
                        [str], List[str]
                    ] = lambda x: apply_bbox_paths_transform(x)

                else:
                    apply_fn: Callable[[str], str] = lambda x: str(
                        Path(x.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH))
                    )

                self.df[p_col] = self.df[p_col].apply(apply_fn)

        ## preprocessing data.
        self.preprocess_label()

        super(ReflacxDataset, self).__init__()

    def preprocess_clinical_df(self,):
        self.encoders_map: Dict[str, LabelEncoder] = {}

        # encode the categorical cols.
        for col in self.clinical_categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders_map[col] = le

    def preprocess_label(self,):
        self.df[self.all_disease_cols] = self.df[self.all_disease_cols].gt(0)

    def load_image_array(self, image_path: str) -> np.ndarray:
        return np.asarray(Image.open(image_path))

    def plot_image_from_array(self, image_array: np.ndarray):
        im = Image.fromarray(image_array)
        im.show()

    def disease_to_idx(self, disease: str) -> int:
        if not disease in self.labels_cols:
            raise Exception("This disease is not the label.")

        if disease == "background":
            return 0

        return self.labels_cols.index(disease) + 1

    def label_idx_to_disease(self, idx: int) -> str:
        if idx == 0:
            return "background"

        if idx > len(self.labels_cols):
            return f"exceed label range :{idx}"

        return self.labels_cols[idx - 1]

    def __len__(self) -> int:
        return len(self.df)

    def generate_bboxes_df(self, ellipse_df: pd.DataFrame,) -> pd.DataFrame:
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

    def set_clinical_features_used(self, clinical_numerical_cols, clinical_categorical_cols):
        self.clinical_numerical_cols = clinical_numerical_cols
        self.clinical_categorical_cols = clinical_categorical_cols

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        # find the df
        data: pd.Series = self.df.iloc[idx]

        img: Image = Image.open(data["image_path"]).convert("RGB")

        ## Get bounding boxes.
        if self.dataset_mode == "unified":
            bboxes_df = pd.concat(
                [self.generate_bboxes_df(pd.read_csv(p)) for p in data["bbox_paths"]],
                axis=0,
            )
        else:
            bboxes_df = self.generate_bboxes_df(
                pd.read_csv(data["anomaly_location_ellipses_path"])
            )
        bboxes = torch.tensor(
            np.array(bboxes_df[self.box_coord_cols], dtype=float)
        )  # x1, y1, x2, y2

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

        img_t, target = self.transforms(img, target)

        if self.with_clinical:
            clinical_num = None
            if not self.clinical_numerical_cols is None and len(self.clinical_numerical_cols) > 0:
                if self.normalise_clinical_num:
                    clinical_num = (
                        torch.tensor(
                            self.clinical_num_norm.transform(
                                np.array([data[self.clinical_numerical_cols]])
                            ),
                            dtype=float,
                        )
                        .float()
                        .squeeze()
                    )
                else:
                    clinical_num = torch.tensor(
                        np.array(data[self.clinical_numerical_cols], dtype=float)
                    ).float()

            clinical_cat = None
            if not self.clinical_categorical_cols is None and len(self.clinical_categorical_cols) > 0:
                clinical_cat = torch.tensor(    
                    np.array(data[self.clinical_categorical_cols], dtype=int)
                )
            return img_t, clinical_num, clinical_cat, target

        return img_t, target

    def prepare_input_from_data(
        self,
        data: Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
            Tuple[torch.Tensor, Dict],
        ],
        device: str,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:

        if self.with_clinical:
            imgs, clinical_num, clinical_cat, targets = data

            imgs = list(img.to(device) for img in imgs)

            if not self.clinical_numerical_cols is None and len(self.clinical_numerical_cols) > 0:
                clinical_num = [t.to(device) for t in clinical_num] 
            if not self.clinical_categorical_cols is None and len(self.clinical_categorical_cols) > 0:
                clinical_cat = [t.to(device) for t in clinical_cat]
            targets = [map_target_to_device(t, device) for t in targets]

            return (imgs, clinical_num, clinical_cat, targets)

        else:
            imgs, targets = data

            imgs = list(img.to(device) for img in imgs)
            targets = [map_target_to_device(t, device) for t in targets]

            return (imgs, targets)

    def get_idxs_from_dicom_id(self, dicom_id: str) -> List[str]:
        return [
            self.df.index.get_loc(i)
            for i in self.df.index[self.df["dicom_id"].eq(dicom_id)]
        ]

    def get_image_path_from_dicom_id(self, dicom_id: str) -> List[str]:
        return self.df[self.df["dicom_id"] == dicom_id].iloc[0]["image_path"]


class OurRadiologsitsDataset(data.Dataset):
    def __init__(self, original_dataset: ReflacxDataset, radiologists_anns: Dict):
        self.original_dataset = original_dataset
        self.radiologists_anns = radiologists_anns
        self.with_clinical = self.original_dataset.with_clinical

        super(OurRadiologsitsDataset, self).__init__()

    def __len__(self) -> int:
        return len(self.radiologists_anns)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        ann: Dict = deepcopy(self.radiologists_anns[idx])

        idx = self.original_dataset.get_idxs_from_dicom_id(ann["dicom_id"])[0]

        data: pd.Series = deepcopy(self.original_dataset[idx])
        # target = data[-1]

        bboxes = ann["boxes"]
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        num_objs = bboxes.shape[0]

        ann["image_id"] = torch.tensor([idx])
        ann["area"] = area
        ann["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        img = Image.open(ann["image_path"]).convert("RGB")
        masks = torch.zeros((num_objs, img.height, img.width), dtype=torch.uint8)
        for i, b in enumerate(bboxes):
            b = b.int()
            masks[i, b[1] : b[3], b[0] : b[2]] = 1
            ann["masks"] = masks

        data = [*data[:-1], ann]

        return data

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
