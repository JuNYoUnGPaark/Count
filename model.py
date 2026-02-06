from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    - Count-only Weakly Supervised Framework
    - Author: JunYoungPark and Myung-Kyu Yi
"""


class ManifoldEncoder(nn.Module):
    """Convolutional encoder that extracts a latent sequence from an input window."""
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        latent_dim: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_dim,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=latent_dim,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        z = self.net(x)
        z = z.transpose(1, 2)
        return z


class ManifoldDecoder(nn.Module):
    """Convolutional decoder that reconstructs the input signal from the latent sequence."""
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        out_channels: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=latent_dim,
                out_channels=hidden_dim,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        zt = z.transpose(1, 2)
        x_hat = self.net(zt)
        return x_hat


class MultiRateHead(nn.Module):
    """Head that predicts a time-varying micro-rate and a K-way phase distribution."""
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        K_max: int,
    ):
        super().__init__()
        self.K_max = int(K_max)
        self.net = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=hidden_dim,
                out_features=1 + self.K_max,
            ),
        )

    def forward(
        self,
        z: torch.Tensor,
        tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(z)
        amp = F.softplus(out[..., 0])
        phase_logits = out[..., 1:]
        phase = F.softmax(phase_logits / float(tau), dim=-1)
        return amp, phase, phase_logits


class KAutoCountModel(nn.Module):
    """Count-only repetition counting model with window-stabilized rate learning."""
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        latent_dim: int,
        K_max: int,
        init_bias: float,
    ):
        super().__init__()
        self.encoder = ManifoldEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.decoder = ManifoldDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_channels=input_channels,
        )
        self.rate_head = MultiRateHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            K_max=K_max,
        )

        self.init_bias = float(init_bias)
        self._init_weights()

    def _init_weights(
        self,
    ) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        with torch.no_grad():
            b = self.rate_head.net[-1].bias
            if b is not None:
                b.zero_()
                b[0].fill_(self.init_bias)

    @staticmethod
    def _masked_mean_time(
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)

        if x.dim() == 2:
            m = mask.to(dtype=x.dtype, device=x.device)
            return (x * m).sum(dim=1) / (m.sum(dim=1) + eps)

        if x.dim() == 3:
            m = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            return (x * m).sum(dim=1) / (m.sum(dim=1) + eps)

        raise ValueError(f"Unsupported dim for masked mean: {x.dim()}")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.encoder(x)
        x_hat = self.decoder(z)

        amp_t, phase_p, _ = self.rate_head(z, tau=tau)
        micro_rate_t = amp_t

        p_bar = self._masked_mean_time(phase_p, mask)
        k_hat = 1.0 / (p_bar.pow(2).sum(dim=1) + 1e-6)

        rep_rate_t = micro_rate_t / (k_hat.unsqueeze(1) + 1e-6)
        if mask is not None:
            rep_rate_t = rep_rate_t * mask

        if mask is None:
            avg_rep_rate = rep_rate_t.mean(dim=1)
        else:
            avg_rep_rate = (rep_rate_t * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)

        aux: Dict[str, torch.Tensor] = {
            "phase_p": phase_p,
            "rep_rate_t": rep_rate_t,
            "k_hat": k_hat,
        }
        return avg_rep_rate, z, x_hat, aux
