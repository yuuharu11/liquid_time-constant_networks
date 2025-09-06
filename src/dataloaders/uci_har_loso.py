# src/dataloaders/uci_har.py


# 未完成！


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .base import SequenceDataset

class UCIHAR_LOSO(SequenceDataset):
    __name__ = "uci_har_loso"

    """
    UCI-HAR (Human Activity Recognition) データセットのための正確なデータローダー。
    公式のtrain/test分割と、Leave-One-Subject-Out (LOSO) 交差検証をサポートします。
    """
    def __init__(self, data_dir, split="train", loso_subject=None, normalize=True, mean=None, std=None):
        self.data_dir = data_dir
        self.split = split
        self.loso_subject = loso_subject
        self.normalize = normalize
        self.signals = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]

        if self.loso_subject is not None:
            # --- LOSOモード ---
            X_all, y_all, subj_all = self._load_all_data()
            if self.split == "train":
                mask = (subj_all != self.loso_subject)
            elif self.split == "test":
                mask = (subj_all == self.loso_subject)
            else: # val
                # 検証データは使わないので空にする
                self.X, self.y = np.array([]), np.array([])
                return 

            self.X = X_all[mask]
            self.y = y_all[mask]
            self.subjects = subj_all[mask]
        else:
            # --- 通常モード (公式のtrain/test分割) ---
            # このフレームワークではvalセットも必要なので、trainの一部をvalとする
            if split == 'val':
                 # trainデータを読み込んで、一部を検証用にする
                X_train_full, y_train_full, _ = self._load_data_from_split("train")
                # 簡単のため、最後の10%を検証用とする
                val_split_idx = int(len(X_train_full) * 0.9)
                self.X, self.y = X_train_full[val_split_idx:], y_train_full[val_split_idx:]
            else:
                 self.X, self.y, self.subjects = self._load_data_from_split(self.split)


        if self.normalize:
            if mean is not None and std is not None:
                self.mean = mean
                self.std = std
            else:
                if self.split == 'train':
                    self.mean = np.mean(self.X, axis=(0, 1), keepdims=True)
                    self.std = np.std(self.X, axis=(0, 1), keepdims=True)
                else:
                    raise ValueError("mean and std must be provided for test/val split when normalize=True")

            self.X = (self.X - self.mean) / (self.std + 1e-8)

        self.X = torch.from_numpy(self.X).float().permute(0, 2, 1)
        self.y = torch.from_numpy(self.y).long()

    def _load_data_from_split(self, split_name):
        base_dir = os.path.join(self.data_dir, "UCI HAR Dataset")
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"'{base_dir}' not found. Download the dataset first.")
        X_list = [np.loadtxt(os.path.join(base_dir, split_name, "Inertial Signals", f"{sig}_{split_name}.txt"), dtype=np.float32) for sig in self.signals]
        X = np.stack(X_list, axis=2)
        y = np.loadtxt(os.path.join(base_dir, split_name, f"y_{split_name}.txt"), dtype=np.int64) - 1
        subj = np.loadtxt(os.path.join(base_dir, split_name, f"subject_{split_name}.txt"), dtype=np.int64)
        return X, y, subj

    def _load_all_data(self):
        X_train, y_train, subj_train = self._load_data_from_split("train")
        X_test, y_test, subj_test = self._load_data_from_split("test")
        return np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)), np.concatenate((subj_train, subj_test))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class UCIHARDataModule(SequenceDataset):
    def __init__(self, loso_subject=None, **kwargs):
        super().__init__(**kwargs)
        self.loso_subject = loso_subject

    def setup(self, stage=None):
        self.train_dataset = UCIHAR_LOSO(data_dir=self.data_dir, split="train", loso_subject=self.loso_subject)
        # LOSOモードでは検証データは使用しない
        self.val_dataset = UCIHAR_LOSO(data_dir=self.data_dir, split="val", loso_subject=self.loso_subject,
                                  mean=self.train_dataset.mean, std=self.train_dataset.std)
        self.test_dataset = UCIHAR_LOSO(data_dir=self.data_dir, split="test", loso_subject=self.loso_subject,
                                   mean=self.train_dataset.mean, std=self.train_dataset.std)