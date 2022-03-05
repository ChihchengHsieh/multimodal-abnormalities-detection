import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torch.autograd import Variable

class REFLACXWithClinicalDataset(data.Dataset):
    
    def __init__(self,
                 image_size=224,
                 clinical_cols=['age', 'gender', 'temperature', 'heartrate', 'resprate',
                                'o2sat', 'sbp', 'dbp', 'pain', 'acuity'],
                 clinical_numerical_cols=[
                     'age', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity'],
                 clinical_categorical_cols=['gender'],
                 #      labels_cols=[
                 # 'Airway wall thickening', 'Atelectasis', 'Consolidation',
                 # 'Enlarged cardiac silhouette', 'Fibrosis',
                 # 'Groundglass opacity', 'Pneumothorax', 'Pulmonary edema',
                 # 'Quality issue', 'Support devices', 'Wide mediastinum',
                 # 'Abnormal mediastinal contour', 'Acute fracture', 'Enlarged hilum',
                 # 'Hiatal hernia', 'High lung volume / emphysema',
                 # 'Interstitial lung disease', 'Lung nodule or mass',
                 # 'Pleural abnormality'
                 #          ],

                 labels_cols=[
                    #  "Support devices",
                     "Enlarged cardiac silhouette",
                     "Atelectasis",
                     "Pleural abnormality",
                     "Consolidation",
                     "Pulmonary edema",
                    #  'Groundglass opacity',
                 ],
                 all_disease_cols=[
            'Airway wall thickening', 'Atelectasis', 'Consolidation',
            'Enlarged cardiac silhouette', 'Fibrosis',
            'Groundglass opacity', 'Pneumothorax', 'Pulmonary edema', 'Wide mediastinum',
            'Abnormal mediastinal contour', 'Acute fracture', 'Enlarged hilum',
            'Hiatal hernia', 'High lung volume / emphysema',
            'Interstitial lung disease', 'Lung nodule or mass',
            'Pleural abnormality'
                 ],
                 horizontal_flip=True,
                 ):

        super(REFLACXWithClinicalDataset, self).__init__()

        self.image_size = image_size
        self.df = pd.read_csv('reflacx_with_clinical.csv', index_col=0)
        self.clinical_cols = clinical_cols
        self.clinical_numerical_cols = clinical_numerical_cols
        self.clinical_categorical_cols = clinical_categorical_cols
        self.labels_cols = labels_cols
        self.all_disease_cols = all_disease_cols
        self.encoders_map = {}

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_transforms_lst = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip() if horizontal_flip else None,
            transforms.ToTensor(),
            normalize,
        ]
        self.train_transform = transforms.Compose(
            [t for t in train_transforms_lst if t])

        self.test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize,
        ])

        self.preprocess_clinical_df()
        self.preprocess_label()
        self.get_weights()

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

    def __getitem__(self, index):
        # find the df
        return self.df.iloc[index]

    def train_collate_fn(self, x):
        return self.collate_fn(x, mode='train')

    def test_collate_fn(self, x):
        return self.collate_fn(x, mode='test')

    def collate_fn(self, data, mode="train"):

        data = pd.DataFrame(data)

        images = [Image.open(path).convert("RGB")
                  for path in data['image_path']]

        label_long_tensor = torch.tensor(
            np.array(data[self.labels_cols])).long()

        clinical_numerical_input = torch.tensor(
            np.array(data[self.clinical_numerical_cols])).float()

        clinical_categorical_input = {}

        for col in self.clinical_categorical_cols:
            clinical_categorical_input[col] = torch.tensor(
                np.array(data[col]))

        images = torch.stack([self.train_transform(
            img) if mode == "train" else self.test_transform(img) for img in images])

        # we will feed the categorical column to the model, so we keep it in dataframe form.
        return images, (clinical_numerical_input, clinical_categorical_input), label_long_tensor

    def __len__(self):
        return len(self.df)

    # def get_weights(self):
    #     p_count = (self.df[self.labels_cols] == 1).sum(axis=0)
    #     self.p_count = p_count
    #     n_count = (self.df[self.labels_cols] == 0).sum(axis=0)
    #     total = p_count + n_count

    #     # invert *opposite* weights to obtain weighted loss
    #     # (positives weighted higher, all weights same across batches, and p_weight + n_weight == 1)
    #     p_weight = n_count / total
    #     n_weight = p_count / total

    #     self.p_weight_loss = Variable(
    #         torch.FloatTensor(p_weight), requires_grad=False)
    #     self.n_weight_loss = Variable(
    #         torch.FloatTensor(n_weight), requires_grad=False)

    #     print("Positive Loss weight:")
    #     print(self.p_weight_loss.data.numpy())
    #     print("Negative Loss weight:")
    #     print(self.n_weight_loss.data.numpy())

    #     n_classes = len(self.labels_cols)

    #     random_loss = sum((p_weight[i] * p_count[i] + n_weight[i] * n_count[i]) *
    #                       -np.log(0.5) / total[i] for i in range(n_classes)) / n_classes
    #     print("Random Loss:")
    #     print(random_loss)

    # def weighted_loss(self, preds, target, device):

    #     weights = (target.type(torch.FloatTensor) * (self.p_weight_loss.expand_as(target)) +
    #                (target == 0).type(torch.FloatTensor) * (self.n_weight_loss.expand_as(target))).to(device)

    #     loss = 0.0
    #     n_classes = len(self.labels_cols)

    #     for i in range(n_classes):
    #         loss += nn.functional.binary_cross_entropy_with_logits(
    #             preds[:, i], target[:, i].float(), weight=weights[:, i])
    #     return loss / n_classes
