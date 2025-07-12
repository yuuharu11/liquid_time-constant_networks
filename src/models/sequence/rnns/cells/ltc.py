# src/models/sequence/rnns/cells/ltc.py

import torch
import torch.nn as nn
import numpy as np
from .basic import CellBase

class LTCCell(CellBase):
    name = 'ltc'  # レジストリに登録するための名前

    def __init__(self, d_input, d_model, ode_solver_unfolds=6, **kwargs):
        self.ode_solver_unfolds = ode_solver_unfolds
        super().__init__(d_input, d_model, **kwargs)
        # TensorFlowの__init__にあった各種パラメータも、必要に応じてここにself変数として追加できます。

    def reset_parameters(self):
        # TensorFlowの_get_variablesメソッドに相当する部分
        # パラメータをPyTorchのnn.Parameterとして定義します
        self._input_size = self.d_input
        self._num_units = self.d_model

        # Sensory Mapping Parameters
        self.sensory_mu = nn.Parameter(torch.empty(self._input_size, self._num_units))
        self.sensory_sigma = nn.Parameter(torch.empty(self._input_size, self._num_units))
        self.sensory_W = nn.Parameter(torch.empty(self._input_size, self._num_units))
        self.sensory_erev = nn.Parameter(torch.empty(self._input_size, self._num_units))
        
        # Recurrent Parameters
        self.mu = nn.Parameter(torch.empty(self._num_units, self._num_units))
        self.sigma = nn.Parameter(torch.empty(self._num_units, self._num_units))
        self.W = nn.Parameter(torch.empty(self._num_units, self._num_units))
        self.erev = nn.Parameter(torch.empty(self._num_units, self._num_units))
        
        # Leak and Membrane Parameters
        self.vleak = nn.Parameter(torch.empty(self._num_units))
        self.gleak = nn.Parameter(torch.empty(self._num_units))
        self.cm_t = nn.Parameter(torch.empty(self._num_units))

        # 初期化処理
        self._initialize_weights()

    def _initialize_weights(self):
        # TensorFlowのinitializerに対応するPyTorchの初期化処理
        nn.init.uniform_(self.sensory_mu, a=0.3, b=0.8)
        nn.init.uniform_(self.sensory_sigma, a=3.0, b=8.0)
        nn.init.uniform_(self.sensory_W, a=0.01, b=1.0)
        sensory_erev_init = torch.tensor(2 * np.random.randint(0, 2, size=(self._input_size, self._num_units)) - 1, dtype=torch.float32)
        self.sensory_erev.data.copy_(sensory_erev_init * 1.0)

        nn.init.uniform_(self.mu, a=0.3, b=0.8)
        nn.init.uniform_(self.sigma, a=3.0, b=8.0)
        nn.init.uniform_(self.W, a=0.01, b=1.0)
        erev_init = torch.tensor(2 * np.random.randint(0, 2, size=(self._num_units, self._num_units)) - 1, dtype=torch.float32)
        self.erev.data.copy_(erev_init * 1.0)

        nn.init.uniform_(self.vleak, a=-0.2, b=0.2)
        nn.init.uniform_(self.gleak, a=1.0, b=1.0)
        nn.init.uniform_(self.cm_t, a=0.5, b=0.5)

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.unsqueeze(-1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def forward(self, inputs, state):
        # PyTorchのforwardメソッド (TensorFlowの__call__に相当)
        v_pre = state

        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        sensory_rev_activation = sensory_w_activation * self.sensory_erev

        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for _ in range(self.ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            rev_activation = w_activation * self.erev

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory
            
            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator

        next_state = v_pre
        output = next_state # LTCではoutputとstateが同じ
        return output, next_state