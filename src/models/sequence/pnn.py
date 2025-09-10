import copy
import torch
import torch.nn as nn
from typing import Any, Dict, List
import src.utils as utils
from src.utils import registry

class PNN(nn.Module):
    """
    Minimal PNN scaffolding.
    - base_model_config: dict config for creating a single-column model (backbone+head).
    - d_output: number of classes / output dim (kept for API compatibility).
    Usage:
        model = PNNSimple(base_model_config, d_output)
        model.add_column(task_id=0)  # add first column
        model.add_column(task_id=1)  # add second column when new task arrives
    """
    def __init__(self, base_model_config: Dict[str, Any], d_output: int):
        super().__init__()
        # Keep a copy of the base config to instantiate new columns
        self.base_model_config = copy.deepcopy(base_model_config)
        self.d_output = d_output

        # Modules
        self.columns: nn.ModuleList = nn.ModuleList()  # list of column modules
        # lateral adapters: list of ModuleList (for each new column, adapters from previous columns)
        self.laterals: List[nn.ModuleList] = []

        # track task ids -> column idx mapping
        self.task_to_col = {}

    def _build_column(self):
        # Instantiate a model column from base config using existing registry utils
        # Expect base_model_config['_name_'] to refer to a backbone (not 'pnn' itself)
        col = utils.instantiate(registry.model, copy.deepcopy(self.base_model_config))
        return col

    def add_column(self, task_id=None, freeze_prev=True):
        """Add a new column for a new task.
        If task_id provided, map it to this new column."""
        col = self._build_column()
        # create lateral adapters from all previous columns to this new column (identity by default)
        adapters = nn.ModuleList()
        for _ in range(len(self.columns)):
            # default adapter: identity; replace with linear/conv as needed
            adapters.append(nn.Identity())
        self.columns.append(col)
        self.laterals.append(adapters)
        col_idx = len(self.columns) - 1
        if task_id is not None:
            self.task_to_col[task_id] = col_idx

        # Freeze previous columns if requested
        if freeze_prev and col_idx > 0:
            for p in self.columns[col_idx - 1].parameters():
                p.requires_grad = False

        return col_idx

    def forward(self, x, **kwargs):
        """
        Forward through the column corresponding to current task.
        kwargs may contain 'task_id' to select column; otherwise use last column.
        """
        task_id = kwargs.get("task_id", None)
        if task_id is None:
            col_idx = len(self.columns) - 1
        else:
            col_idx = self.task_to_col.get(task_id, len(self.columns) - 1)

        # simple forward: run target column only, optionally incorporate laterals
        # (more advanced: combine previous columns' intermediate features via adapters)
        col = self.columns[col_idx]
        return col(x)
# ...existing code...