import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch.nn.utils.rnn import pack_padded_sequence

from typing import Dict, Any, Optional

class TemporalEncoderGRU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int = 128, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(d_hidden)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, T, D) padded
        lengths: (B,) actual lengths (on CPU or GPU; we'll .cpu() for packing)
        Returns: (B, d_hidden) embedding for MLP input
        """
        # Pack to ignore padding; enforce_sorted=False so we don't need to sort the batch
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # h_n: (num_layers, B, d_hidden)

        # last layer hidden state for each sequence = last valid timestep
        h_last = h_n[-1]  # (B, d_hidden)
        return self.layer_norm(h_last)


class Predictor(nn.Module):
    def __init__(self, config: Dict[str, Any], K: int, Dx_total: int, Da: int, Ds: int, n_users: int):
        super().__init__()
        mcfg = config["model"]
        self.K = K
        self.lowrank_r = int(mcfg["lowrank_r"])

        self.encoder = TemporalEncoderGRU(d_in=K + Dx_total,
                                          d_hidden=mcfg["d_hidden"],
                                          num_layers=mcfg["enc_layers"],
                                          dropout=mcfg["dropout"])

        self.user_emb = None
        uemb_dim = int(mcfg["user_embed_dim"])
        if uemb_dim > 0:
            self.user_emb = nn.Embedding(n_users, uemb_dim)

        head_in = mcfg["d_hidden"] + Da + Ds + (uemb_dim if uemb_dim > 0 else 0)
        self.trunk = nn.Sequential(
            nn.Linear(head_in, mcfg["trunk_hidden"]), nn.ReLU(), nn.Dropout(mcfg["dropout"]),
            nn.Linear(mcfg["trunk_hidden"], mcfg["trunk_hidden"]), nn.ReLU(), nn.Dropout(mcfg["dropout"])
        )

        self.mu_head    = nn.Linear(mcfg["trunk_hidden"], K)
        self.scale_head = nn.Linear(mcfg["trunk_hidden"], K)
        self.B_head     = nn.Linear(mcfg["trunk_hidden"], K * self.lowrank_r) if self.lowrank_r > 0 else None

    def forward(self, y_hist, x_hist, a_now, s_static, user_idx):
        seq = torch.cat([y_hist, x_hist], dim=-1)              # [B, W, K+Dx_total]
        h = self.encoder(seq)
        parts = [h, a_now, s_static]
        if self.user_emb is not None:
            parts.append(self.user_emb(user_idx))
        z = self.trunk(torch.cat(parts, dim=-1))

        mu = self.mu_head(z)
        sigma = F.softplus(self.scale_head(z)) + 1e-4

        g_mu = mu.mean(dim=-1)
        diag_term = (sigma ** 2).sum(dim=-1) / (self.K ** 2)
        if self.B_head is not None:
            B = self.B_head(z).view(-1, self.K, self.lowrank_r)
            ones = torch.ones(self.K, device=B.device)
            bsum = torch.einsum("bkr,k->br", B, ones)
            cross_term = (bsum ** 2).sum(dim=-1) / (self.K ** 2)
            g_var = diag_term + cross_term
        else:
            B, g_var = None, diag_term

        return {"mu": mu, "sigma": sigma, "g_mu": g_mu, "g_var": g_var, "B": B}


@dataclass
class LossCfg:
    lambda_scores: float = 1.0
    lambda_g: float = 0.2
    reduction: str = "mean"


class PredictorLossMasked(nn.Module):
    def __init__(self, cfg: LossCfg):
        super().__init__()
        assert cfg.reduction in ("mean", "sum")
        self.cfg = cfg

    def forward(self, preds, y_true, m_true):
        mu, sigma = preds["mu"], torch.clamp(preds["sigma"], min=1e-6)
        log_sigma = torch.log(sigma)
        nll_elem = 0.5 * (((y_true - mu) / sigma) ** 2 + 2.0 * log_sigma)
        nll_masked = nll_elem * m_true
        denom = m_true.sum().clamp_min(1e-8)
        loss_scores = nll_masked.sum() / denom if self.cfg.reduction == "mean" else nll_masked.sum()

        m_sum = m_true.sum(dim=-1).clamp_min(1.0)
        g_true_obs = (y_true * m_true).sum(dim=-1) / m_sum
        loss_g = F.mse_loss(preds["g_mu"], g_true_obs, reduction=self.cfg.reduction)

        loss = self.cfg.lambda_scores * loss_scores + self.cfg.lambda_g * loss_g
        metrics = {
            "loss_scores": loss_scores.detach(),
            "loss_g": loss_g.detach(),
            "mae": (torch.abs(mu - y_true) * m_true).sum() / denom,
            "g_mae": torch.abs(preds["g_mu"] - g_true_obs).mean(),
        }
        return loss, metrics


class Trainer:
    def __init__(self, config: Dict[str, Any], model: Predictor, loss_fn: PredictorLossMasked, device: Optional[str] = None):
        self.cfg = config
        self.model = model
        self.loss_fn = loss_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        tcfg = config["train"]
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=tcfg["epochs"])
        self.grad_clip = tcfg["grad_clip"]

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def train_one_epoch(self, dl: DataLoader) -> Dict[str, float]:
        self.model.train()
        agg = {"loss": 0.0, "mae": 0.0, "g_mae": 0.0}; n = 0
        for batch in dl:
            b = self._to_device(batch)
            preds = self.model(b["y_hist"], b["x_hist"], b["a_now"], b["s_static"], b["user_idx"])
            loss, metrics = self.loss_fn(preds, b["y_next"], b["m_next"])
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optim.step()
            agg["loss"] += float(loss.item())
            agg["mae"]  += float(metrics["mae"].item())
            agg["g_mae"]+= float(metrics["g_mae"].item())
            n += 1
        for k in agg: agg[k] /= max(n, 1)
        self.scheduler.step()
        return agg

    @torch.no_grad()
    def evaluate(self, dl: DataLoader) -> Dict[str, float]:
        self.model.eval()
        agg = {"loss": 0.0, "mae": 0.0, "g_mae": 0.0}; n = 0
        for batch in dl:
            b = self._to_device(batch)
            preds = self.model(b["y_hist"], b["x_hist"], b["a_now"], b["s_static"], b["user_idx"])
            loss, metrics = self.loss_fn(preds, b["y_next"], b["m_next"])
            agg["loss"] += float(loss.item())
            agg["mae"]  += float(metrics["mae"].item())
            agg["g_mae"]+= float(metrics["g_mae"].item())
            n += 1
        for k in agg: agg[k] /= max(n, 1)
        return agg