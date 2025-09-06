import os
import numpy as np
import torch
from torch.utils.data import Dataset

class UCIHAR(Dataset):
    def __init__(self, data_dir, split="train", subjects=None, window_size=128, overlap=0.5, normalize=True):
        """
        UCI-HAR Dataset Loader

        Args:
            data_dir (str or Path): データセットディレクトリ
            split (str): "train" or "test"
            subjects (list[int]): 使用する被験者 ID リスト
            window_size (int): ウィンドウサイズ
            overlap (float): ウィンドウオーバーラップ率
            normalize (bool): True の場合チャネルごとに標準化
        """
        self.data_dir = data_dir
        self.split = split
        self.window_size = window_size
        self.overlap = overlap
        self.normalize = normalize
        self.signals = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]

        # データ読み込み
        self.X, self.y, self.subjects = self._load_data(subjects)

        # 標準化
        if normalize and split=="train":
            self.mean = self.X.mean(axis=(0,1), keepdims=True)
            self.std = self.X.std(axis=(0,1), keepdims=True)
            self.X = (self.X - self.mean) / (self.std + 1e-8)
        elif normalize and split=="test":
            self.X = (self.X - self.mean) / (self.std + 1e-8)

        # ウィンドウ切り出し
        self.X_windows, self.y_windows = self._create_windows()

    def _load_data(self, subjects):
        """
        各信号を読み込み、結合
        """
        base_dir = os.path.join(self.data_dir, "UCI HAR Dataset")
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"{base_dir} not found. Download the dataset first.")

        # データ読み込み
        X_list = []
        for sig in self.signals:
            path = os.path.join(base_dir, self.split, "Inertial Signals", f"{sig}_{self.split}.txt")
            X_list.append(np.loadtxt(path, dtype=np.float32))
        X = np.stack(X_list, axis=2)  # shape: (num_samples, 128, 9)

        # ラベル読み込み
        y_path = os.path.join(base_dir, self.split, f"y_{self.split}.txt")
        y = np.loadtxt(y_path, dtype=np.int32) - 1  # 0-indexed

        # 被験者 ID 読み込み
        subj_path = os.path.join(base_dir, self.split, f"subject_{self.split}.txt")
        subj = np.loadtxt(subj_path, dtype=np.int32)

        # 被験者選択
        if subjects is not None:
            mask = np.isin(subj, subjects)
            X = X[mask]
            y = y[mask]
            subj = subj[mask]

        return X, y, subj

    def _create_windows(self):
        """
        50%オーバーラップでウィンドウ化
        """
        step = int(self.window_size * (1 - self.overlap))
        X_w, y_w = [], []

        for start in range(0, self.X.shape[1] - self.window_size + 1, step):
            X_w.append(self.X[:, start:start+self.window_size, :])
            y_w.append(self.y)  # ウィンドウのラベルは元ラベルそのまま

        X_w = np.concatenate(X_w, axis=0)
        y_w = np.concatenate(y_w, axis=0)
        return torch.from_numpy(X_w).float(), torch.from_numpy(y_w).long()

    def __len__(self):
        return len(self.y_windows)

    def __getitem__(self, idx):
        return self.X_windows[idx], self.y_windows[idx]
