import os
import numpy as np
import torch
from .base import SequenceDataset
from src.dataloaders.base import default_data_path

class UCIHAR_CIL(SequenceDataset):
    _name_ = "uci_har_cil"
    d_input = 9
    d_output = 6
    L = 128

    def __init__(self, data_dir=None, val_split=0.2, seed=42, task_id=0, **kwargs):
        self.data_dir = data_dir
        self.val_split = val_split
        self.seed = seed
        self.task_id = task_id
        
        # --- CILシナリオの定義 ---
        # 各タスクで学習する「新しい」クラスのリスト
        self.tasks = [
            [3, 4, 5],  # Task 0: SITTING, STANDING, LAYING (静的活動)
            [0],        # Task 1: WALKING (動的活動の追加)
            [1, 2]      # Task 2: WALKING_UPSTAIRS, WALKING_DOWNSTAIRS (類似活動)
        ]
        # UCI-HAR Labels: 0-WALKING, 1-WALKING_UPSTAIRS, 2-WALKING_DOWNSTAIRS, 3-SITTING, 4-STANDING, 5-LAYING

        # 現在のタスクで可視（学習・評価対象）となる全クラスを決定
        self.visible_classes = []
        for i in range(task_id + 1):
            self.visible_classes.extend(self.tasks[i])

        # 必ずsetupを呼び出す
        self.setup()

    def setup(self):
        self.data_dir = self.data_dir or default_data_path / "uci_har"
        data_path = self.data_dir / "UCI HAR Dataset"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found in {self.data_dir}.")

        # --- ステップ1: 公式のtrain/testデータを別々にロード ---
        X_train_full = self._load_X(data_path / "train/Inertial Signals/", "train")
        y_train_full = self._load_y(data_path / "train/y_train.txt")
        X_test_full = self._load_X(data_path / "test/Inertial Signals/", "test")
        y_test_full = self._load_y(data_path / "test/y_test.txt")

        # --- ステップ2: train/testそれぞれで、クラスのフィルタリングを行う ---
        print(f"--- CIL Task {self.task_id}: Filtering for classes {self.visible_classes} ---")

        # 訓練データから、見えるべきクラスだけを抽出
        train_mask = np.isin(y_train_full, self.visible_classes)
        X_train_visible, y_train_visible = X_train_full[train_mask], y_train_full[train_mask]

        # テストデータから、見えるべきクラスだけを抽出
        test_mask = np.isin(y_test_full, self.visible_classes)
        X_test_visible, y_test_visible = X_test_full[test_mask], y_test_full[test_mask]

        # --- ステップ3: データの分割と準備 ---

        # フィルタリング後の訓練データを、さらに学習用と検証用に分割
        np.random.seed(self.seed)
        n_train_samples = len(X_train_visible)
        indices = np.random.permutation(n_train_samples)
        val_size = int(n_train_samples * self.val_split)
        train_indices = indices[:-val_size]
        val_indices = indices[-val_size:]

        X_train_split = X_train_visible[train_indices]
        y_train_split = y_train_visible[train_indices]
        
        X_val_split = X_train_visible[val_indices]
        y_val_split = y_train_visible[val_indices]

        # 正規化のための統計量は、常にその時点の「学習用」データからのみ計算
        mean = np.mean(X_train_split, axis=(0, 1), keepdims=True)
        std = np.std(X_train_split, axis=(0, 1), keepdims=True)
        
        # 全てのデータセットを、学習データの統計量で正規化
        X_train_norm = (X_train_split - mean) / (std + 1e-8)
        X_val_norm = (X_val_split - mean) / (std + 1e-8)
        X_test_norm = (X_test_visible - mean) / (std + 1e-8)

        # TensorDataset に変換
        self.dataset_train = self._to_tensor_dataset(X_train_norm, y_train_split)
        self.dataset_val = self._to_tensor_dataset(X_val_norm, y_val_split)
        self.dataset_test = self._to_tensor_dataset(X_test_norm, y_test_visible)

    def _to_tensor_dataset(self, x, y):
        x_tensor = torch.from_numpy(x).float().permute(0, 2, 1)
        y_tensor = torch.from_numpy(y).long()
        return torch.utils.data.TensorDataset(x_tensor, y_tensor)

    def _load_X(self, path, split="train"):
        signals = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y", "total_acc_z"]
        X = [np.loadtxt(path / f"{sig}_{split}.txt", dtype=np.float32) for sig in signals]
        return np.transpose(np.array(X), (1, 2, 0))

    def _load_y(self, path):
        return np.loadtxt(path, dtype=np.int32) - 1