import os
import numpy as np
import torch
from .base import SequenceDataset
from src.dataloaders.base import default_data_path

class UCIHAR_DIL(SequenceDataset):
    _name_ = "uci_har_dil"
    d_input = 9
    d_output = 6
    l_output = 0
    L = 128

    @property
    def init_defaults(self):
        return {
            "val_split": 0.2, 
            "seed": 42, 
            "normalize": True,
            "task_id": 0,          # DILタスクID (0: total_acc, 1: body_acc, 2: body_gyro)
            "noise_level": 0.0     # ノイズの強さ (0.0はノイズなし)
        }

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "uci_har"
        
        data_path = self.data_dir / "UCI HAR Dataset"
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"UCI HAR Dataset not found in {self.data_dir}. "
                f"Please download the dataset first."
            )

        # 常にクリーンなデータをロード
        X_train_clean = self._load_X(data_path / "train/Inertial Signals/", "train")
        y_train = self._load_y(data_path / "train/y_train.txt")
        X_test_clean = self._load_X(data_path / "test/Inertial Signals/", "test")
        y_test = self._load_y(data_path / "test/y_test.txt")

        # 正規化
        if getattr(self, "normalize", True):
            mean = np.mean(X_train_clean, axis=(0, 1), keepdims=True)
            std = np.std(X_train_clean, axis=(0, 1), keepdims=True)
            X_train_clean = (X_train_clean - mean) / (std + 1e-8)
            X_test_clean = (X_test_clean - mean) / (std + 1e-8)
        
        # --- DILタスク：指定されたセンサーグループにノイズを追加 ---
        if self.noise_level > 0:
            print(f"Applying Gaussian noise (level: {self.noise_level}) to sensor group {self.task_id}...")
            # データの標準偏差を基準にノイズの大きさを決める
            data_std = np.std(X_train_clean)
            noise_amplitude = data_std * self.noise_level

            # train と test の両方に同じノイズを追加
            X_train = self._apply_noise(X_train_clean, self.task_id, noise_amplitude)
            X_test = self._apply_noise(X_test_clean, self.task_id, noise_amplitude)
        else:
            # ノイズレベルが0なら、そのままクリーンなデータを使用
            X_train = X_train_clean
            X_test = X_test_clean

        # TensorDatasetに変換
        self.dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        self.dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        
        # 訓練データを学習用と検証用に分割
        self.split_train_val(self.val_split)

    def _apply_noise(self, data, task_id, noise_amplitude):
        """特定のセンサーグループのチャネルにガウシアンノイズを追加する"""
        noisy_data = np.copy(data)
        
        # センサーグループとチャネルのインデックスを対応付ける
        # total_acc: [6, 7, 8], body_acc: [0, 1, 2], body_gyro: [3, 4, 5]
        sensor_channels = {
            0: [6, 7, 8], # total_acc
            1: [0, 1, 2], # body_acc
            2: [3, 4, 5],  # body_gyro
            3: [0, 1, 2, 3, 4, 5], # body_acc + body_gyro
            4: [0, 1, 2, 6, 7, 8], # body_acc + total_acc
            5: [3, 4, 5, 6, 7, 8], # body_gyro + total_acc
            6: [0, 1, 2, 3, 4, 5, 6, 7, 8], # all sensors
        }
        
        channels_to_corrupt = sensor_channels.get(task_id)
        if channels_to_corrupt is None:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 0, 1, 2, 3, 4, 5, or 6.")

        # 指定されたチャネルにのみノイズを印加
        noise = np.random.normal(0, noise_amplitude, noisy_data[:, :, channels_to_corrupt].shape)
        noisy_data[:, :, channels_to_corrupt] += noise
        
        return noisy_data

    def _load_X(self, path, split="train"):
        signals = [
            "body_acc_x", "body_acc_y", "body_acc_z",      # Channels 0, 1, 2
            "body_gyro_x", "body_gyro_y", "body_gyro_z",   # Channels 3, 4, 5
            "total_acc_x", "total_acc_y", "total_acc_z",   # Channels 6, 7, 8
        ]
        X = [np.loadtxt(path / f"{sig}_{split}.txt", dtype=np.float32) for sig in signals]
        return np.transpose(np.array(X), (1, 2, 0))

    def _load_y(self, path):
        return np.loadtxt(path, dtype=np.int32) - 1