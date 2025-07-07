import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

class BaseDataModule(pl.LightningDataModule):
    """
    データローダーの共通ロジックをまとめた基底クラス。
    具体的なデータセットを扱うクラスは、これを継承して作成する。
    """
    def __init__(
        self,
        loader_cfg: DictConfig, # conf/loader/default.yaml の設定を受け取る
        # dataset_cfg: DictConfig など、必要に応じて他の設定も受け取る
    ):
        super().__init__()
        # 受け取った設定をハイパーパラメータとして保存
        # self.hparams.loader_cfg.batch_size のようにアクセスできる
        self.save_hyperparameters()

        # 各データセットはsetup()メソッドで割り当てる
        self.dataset_train: Dataset | None = None
        self.dataset_val: Dataset | None = None
        self.dataset_test: Dataset | None = None

    def prepare_data(self):
        """
        データのダウンロードなど、一度だけ行えばよい処理をここに記述する。
        （例: MNISTの初回ダウンロード）
        このメソッドはメインプロセスでのみ実行される。
        """
        # この基底クラスでは何もしない。継承先で実装する。
        pass

    def setup(self, stage: str = None):
        """
        データセットの分割や、transformの適用などを行う。
        このメソッドは各GPUプロセスで実行される。
        """
        # このメソッドは、具体的なデータセットを扱う子クラスで必ず実装する必要がある
        raise NotImplementedError("`setup()` must be implemented in a subclass.")

    def train_dataloader(self) -> DataLoader:
        """訓練データ用のDataLoaderを返す"""
        if self.dataset_train is None:
            raise ValueError("Training dataset is not available. `setup()` must be called first.")
        
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.hparams.loader_cfg.batch_size,
            shuffle=self.hparams.loader_cfg.get("shuffle", True),
            num_workers=self.hparams.loader_cfg.num_workers,
            pin_memory=self.hparams.loader_cfg.pin_memory,
            drop_last=self.hparams.loader_cfg.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        """検証データ用のDataLoaderを返す"""
        if self.dataset_val is None:
            raise ValueError("Validation dataset is not available.")

        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.hparams.loader_cfg.batch_size,
            shuffle=False, # 検証・テストではシャッフルしない
            num_workers=self.hparams.loader_cfg.num_workers,
            pin_memory=self.hparams.loader_cfg.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """テストデータ用のDataLoaderを返す"""
        if self.dataset_test is None:
            raise ValueError("Test dataset is not available.")

        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.hparams.loader_cfg.batch_size,
            shuffle=False,
            num_workers=self.hparams.loader_cfg.num_workers,
            pin_memory=self.hparams.loader_cfg.pin_memory,
        )