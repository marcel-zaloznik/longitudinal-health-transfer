"""GRU-D model and masked objective with fixed dynamic feature positions."""
import torch
import torch.nn as nn

class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, X_mean, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_gamma_x = nn.Parameter(torch.zeros(input_size))
        self.b_gamma_x = nn.Parameter(torch.zeros(input_size))
        self.W_gamma_h = nn.Parameter(torch.zeros(hidden_size, input_size))
        self.b_gamma_h = nn.Parameter(torch.zeros(hidden_size))

        gate_input_dim = input_size + hidden_size + input_size
        self.z_gate = nn.Linear(gate_input_dim, hidden_size)
        self.r_gate = nn.Linear(gate_input_dim, hidden_size)
        self.h_gate = nn.Linear(gate_input_dim, hidden_size)

        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32))

        nn.init.xavier_uniform_(self.z_gate.weight)
        nn.init.zeros_(self.z_gate.bias)
        nn.init.xavier_uniform_(self.r_gate.weight)
        nn.init.zeros_(self.r_gate.bias)
        nn.init.xavier_uniform_(self.h_gate.weight)
        nn.init.zeros_(self.h_gate.bias)

        self.output_layer = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, X, X_last, M, Delta, seq_mask=None):
        batch_size, seq_len, _ = X.size()
        device = X.device

        h = torch.zeros(batch_size, self.hidden_size, device=device)

        if seq_mask is None:
            seq_mask = torch.ones(batch_size, seq_len, device=device)

        for t in range(seq_len):
            mask_t = seq_mask[:, t].unsqueeze(1)

            x_t = X[:, t, :]
            x_last_t = X_last[:, t, :]
            m_t = M[:, t, :]
            delta_t = Delta[:, t, :]

            h_prev = h

            gamma_x = torch.exp(
                -torch.clamp(self.W_gamma_x * delta_t + self.b_gamma_x, min=0.0)
            )

            gamma_h = torch.exp(
                -torch.clamp(delta_t.matmul(self.W_gamma_h.t()) + self.b_gamma_h, min=0.0)
            )

            x_mean = self.X_mean.unsqueeze(0).expand_as(x_t)
            x_hat = m_t * x_t + (1.0 - m_t) * (
                gamma_x * x_last_t + (1.0 - gamma_x) * x_mean
            )

            h_decay = gamma_h * h_prev

            combined = torch.cat([x_hat, h_decay, m_t], dim=1)
            z = torch.sigmoid(self.z_gate(combined))
            r = torch.sigmoid(self.r_gate(combined))

            combined_r = torch.cat([x_hat, r * h_decay, m_t], dim=1)
            h_tilde = torch.tanh(self.h_gate(combined_r))

            h_new = (1.0 - z) * h_decay + z * h_tilde
            h = mask_t * h_new + (1.0 - mask_t) * h_prev

        y_hat = self.output_layer(h)
        return y_hat

def target_balanced_masked_mse(y_pred, y_true, y_mask, target_weights):
    """
    Target-balanced masked MSE.

    Prevents dense targets from dominating sparse targets.
    """
    se = (y_pred - y_true) ** 2
    weighted_mask = y_mask * target_weights.view(1, -1)
    denom = weighted_mask.sum().clamp_min(1e-8)
    return (se * weighted_mask).sum() / denom
