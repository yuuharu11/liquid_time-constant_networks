import os
import zipfile
import requests
import numpy as np
import torch
from .base import SequenceDataset
from ..utils.config import default_data_path

class UCIHAR(SequenceDataset):
    _name_ = "uci_har"
    d_input = 9  # 9軸のセンサーデータ
    d_output = 6   # 6クラスの行動
    l_output = 0
    L = 128      # 1つのサンプルは128ステップの時系列

    @property
    def init_defaults(self):
        return {"val_split": 0.2, "seed": 42}

    def setup(self):
        # データのパスをDockerコンテナ内の構造に合わせる
        self.data_dir = self.data_dir or default_data_path / "uci_har"
        
        # データの読み込み
        data_path = self.data_dir / "UCI HAR Dataset"
        # もしデータが存在しない場合はエラーメッセージを出す
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"UCI HAR Dataset not found in {self.data_dir}. "
                f"Please run the download commands in the terminal first."
            )

        X_train_path = data_path / "train/Inertial Signals/"
        y_train_path = data_path / "train/y_train.txt"
        X_test_path = data_path / "test/Inertial Signals/"
        y_test_path = data_path / "test/y_test.txt"

        X_train = self._load_X(X_train_path, "train")
        y_train = self._load_y(y_train_path)
        X_test = self._load_X(X_test_path, "test")
        y_test = self._load_y(y_test_path)

        # PyTorchのテンソルに変換
        self.dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        self.dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        
        self.split_train_val(self.val_split)

    def _load_X(self, path, split="train"):
        signals = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]
        X = []
        for sig_name in signals:
            file_name = f"{sig_name}_{split}.txt"
            file_path = os.path.join(path, file_name)
            signal = np.loadtxt(file_path, dtype=np.float32)
            X.append(signal)
        # (チャンネル数, サンプル数, シーケンス長) -> (サンプル数, シーケンス長, チャンネル数)
        return np.transpose(np.array(X), (1, 2, 0))

    def _load_y(self, path):
        # ラベルを0-indexedにする (元のラベルは1-6)
        return np.loadtxt(path, dtype=np.int32) - 1