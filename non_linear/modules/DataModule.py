import json
import sqlite3

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.transforms import transforms

from utils.utils import crop_center


class AttackDataset(Dataset):
    def __init__(self, db_file, img_size, crop_size, do_crop, transform):
        """
        Initializes the dataset used for the DL attacks. Connects to the db that contains the CRPs and provides access
        to the data. All data is transformed by the provided function before being returned.

        :param db_file: full path to the db file that contains the data
        :param img_size: size of the responses
        :param crop_size: size to which the responses will be cropped if decided to
        :param do_crop: whether to crop the responses
        :param transform: transformations that are applied to the responses before feeding them to the model
        """
        try:
            conn = sqlite3.connect(db_file)
        except sqlite3.Error as e:
            print("Error opening db file ", db_file)
            print(e)

        cursor = conn.cursor()
        self.cursor = cursor
        self.img_size = img_size
        self.crop_size = crop_size
        self.crop = do_crop
        self.transform = transform
        self.table_name = cursor.execute("select name from sqlite_master where type = 'table';").fetchone()[0]

    def __len__(self):
        return self.cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]

    def __getitem__(self, idx):
        db_id = idx + 1
        challenge, response = self.cursor.execute(
            f"SELECT challenge, response FROM {self.table_name} WHERE id = {db_id}").fetchone()

        challenge = torch.tensor([int(bit) for bit in challenge], dtype=torch.float)
        response = np.array(json.loads(response)).astype(np.float32)
        response = response.reshape((self.img_size, self.img_size))
        if self.crop:
            response = crop_center(response, self.crop_size, self.img_size)

        response = self.transform(response).float()
        return challenge, response


class AttackDataModule(LightningDataModule):
    def __init__(self, batch_size, db_file, img_size, crop_size, do_crop):
        """
        Initializes the data module used for the DL attacks. The data is split into a training and test set using a
        90/10 split. Provides two dataloaders to access both datasets.

        :param batch_size: batch size for the returned batches
        :param db_file: full path to the db file that contains the data
        :param img_size: size of the responses
        :param crop_size: size to which the responses will be cropped if decided to
        :param do_crop: whether to crop the responses
        """
        super().__init__()
        self.batch_size = batch_size
        self.denormalize = transforms.Normalize(mean=[-1.0], std=[2.0])
        self.db_file = db_file
        self.img_size = img_size
        self.crop_size = crop_size
        self.crop = do_crop
        self.train_kwargs = {"batch_size": self.batch_size, "num_workers": 4, "pin_memory": True}
        self.test_kwargs = {"batch_size": self.batch_size, "num_workers": 4, "pin_memory": True}
        self.dataset = None
        self.train_sampler = None
        self.test_sampler = None

    def setup(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.dataset = AttackDataset(self.db_file, self.img_size, self.crop_size, self.crop, transform=transform)
        data_size = len(self.dataset)
        train_indices = list(range(0, int(data_size * 0.9)))
        test_indices = list(range(int(data_size * 0.9), data_size))
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def train_dataloader(self):
        return DataLoader(self.dataset, sampler=self.train_sampler, **self.train_kwargs)

    def test_dataloader(self):
        return DataLoader(self.dataset, sampler=self.test_sampler, **self.test_kwargs)
