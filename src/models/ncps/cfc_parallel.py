import torch
from torch import nn
from typing import Optional, Union
import ncps
from .cells import CfCCell, WiredCfCCell
from .lstm import LSTMCell
from src.models.wirings import AutoNCP
from ..sequence import SequenceModule
from .cells import ParallelCfCCell as CfcCell

class Cfc(SequenceModule):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_feature,
        hparams,
        return_sequences=False,
        use_mixed=False,
        use_ltc=False,
    ):
        super(Cfc, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences

        self.rnn_cell = CfcCell(in_features, hidden_size, hparams)
        self.use_mixed = use_mixed
        if self.use_mixed:
            self.lstm = LSTMCell(in_features, hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.out_feature)

    def forward(self, x, timespans=None, mask=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        true_in_features = x.size(2)
        h_state = torch.zeros((batch_size, self.hidden_size), device=device)
        if self.use_mixed:
            c_state = torch.zeros((batch_size, self.hidden_size), device=device)
        output_sequence = []
        if mask is not None:
            forwarded_output = torch.zeros(
                (batch_size, self.out_feature), device=device
            )
            forwarded_input = torch.zeros((batch_size, true_in_features), device=device)
            time_since_update = torch.zeros(
                (batch_size, true_in_features), device=device
            )
        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].squeeze()
            if mask is not None:
                if mask.size(-1) == true_in_features:
                    forwarded_input = (
                        mask[:, t] * inputs + (1 - mask[:, t]) * forwarded_input
                    )
                    time_since_update = (ts.view(batch_size, 1) + time_since_update) * (
                        1 - mask[:, t]
                    )
                else:
                    forwarded_input = inputs
                if (
                    true_in_features * 2 < self.in_features
                    and mask.size(-1) == true_in_features
                ):
                    # we have 3x in-features
                    inputs = torch.cat(
                        (forwarded_input, time_since_update, mask[:, t]), dim=1
                    )
                else:
                    # we have 2x in-feature
                    inputs = torch.cat((forwarded_input, mask[:, t]), dim=1)
            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if mask is not None:
                cur_mask, _ = torch.max(mask[:, t], dim=1)
                cur_mask = cur_mask.view(batch_size, 1)
                current_output = self.fc(h_state)
                forwarded_output = (
                    cur_mask * current_output + (1.0 - cur_mask) * forwarded_output
                )
            if self.return_sequences:
                output_sequence.append(self.fc(h_state))

        if self.return_sequences:
            readout = torch.stack(output_sequence, dim=1)
        elif mask is not None:
            readout = forwarded_output
        else:
            readout = self.fc(h_state)
        return readout
