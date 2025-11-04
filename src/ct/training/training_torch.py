
import torch
import numpy as np
from tqdm import tqdm
from typing import Callable, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# =====================================================
# Custom masked MSE via torch.autograd.Function
# =====================================================

from typing import Optional, Tuple

class _MaskedMSEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Forward: MSE computed only over positions where mask==1, normalized by mask.sum()."""
        ctx.save_for_backward(input, target, mask)
        ctx.eps = eps
        diff = (input - target) ** 2              # (B, 14)
        masked = diff * mask                      # zero out missing
        denom = torch.clamp(mask.sum(), min=eps)  # scalar
        return masked.sum() / denom

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx, *grad_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        (grad_output,) = grad_outputs  # single-output function
        input, target, mask = ctx.saved_tensors
        denom = torch.clamp(mask.sum(), min=ctx.eps)
        grad_input = 2.0 * (input - target) * mask / denom
        grad_input = grad_input * grad_output  # chain rule
        # grads for (input, target, mask, eps)
        return grad_input, None, None, None  # pyright: ignore[reportReturnType]


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    return _MaskedMSEFunction.apply(pred, target, mask)


# =====================================================
# Public loss API
# =====================================================

def MSE(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error loss."""
    return torch.mean((pred - target) ** 2)

def MAE(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error loss."""
    return torch.mean(torch.abs(pred - target))

def get_loss_function(loss_function_name: str) -> Callable:
    """
    Return a callable loss function.
    The returned callable has attribute `requires_mask: bool` so training/eval know whether to build masks.
    """
    name = loss_function_name.lower()
    if name == "mse":
        fn = MSE
        setattr(fn, "requires_mask", False)
        return fn
    if name == "mae":
        fn = MAE
        setattr(fn, "requires_mask", False)
        return fn
    if name in ("masked_mse", "masked-mse", "masked"):
        def _call(pred: torch.Tensor, target: torch.Tensor, *, mask: Optional[torch.Tensor] = None):
            if mask is None:
                raise ValueError("masked_mse requires a mask tensor of shape (B, 14).");
            return masked_mse_loss(pred, target, mask)
        setattr(_call, "requires_mask", True)
        return _call
    raise ValueError(f"Unsupported loss function: {loss_function_name}")


# =====================================================
# Mask utilities (from inputs)
# =====================================================

def compute_non_missing_mask_from_inputs(inputs: torch.Tensor) -> torch.Tensor:
    """
    Compute (B,14) mask from inputs (B,42).
    For each domain j, pair = (inputs[:, 14+2*j], inputs[:, 14+2*j+1]).
    Mark valid=1 if pair != (0,0) and != (1,1), else 0.
    Returns a float mask (same dtype/device as inputs).
    """
    if inputs.dim() != 2 or inputs.size(1) != 42:
        raise ValueError("Expected inputs shape (B, 42).");
    B = inputs.size(0)
    mask = torch.ones((B, 14), dtype=inputs.dtype, device=inputs.device)
    for j in range(14):
        base = 14 + 2*j
        a = inputs[:, base]
        b = inputs[:, base + 1]
        valid = ~((a == 0) & (b == 0)) & ~((a == 1) & (b == 1))
        mask[:, j] = valid.to(inputs.dtype)
    return mask


# =====================================================
# Data helpers expected by 03_train_predictor.py
# =====================================================

class custom_dataset(Dataset):
    """Simple dataset wrapper for (X, y) tensors."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Return a DataLoader."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=bool(shuffle))


def split_train_test(data: np.ndarray, ratio: float = 0.2, n_samples: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Split numpy array into (train, test) with given ratio. Optionally subsample first."""
    if n_samples is not None and n_samples > 0 and n_samples < data.shape[0]:
        data = data[:n_samples].copy()
    train, test = train_test_split(data, test_size=ratio, shuffle=True, random_state=42)
    return train, test


def split_input_target(data: np.ndarray, dims: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split full data (numpy) into input and target tensors.
    Inputs: first `dims` columns. Targets: remaining columns.
    """
    input_data = data[:, :dims]
    target_data = data[:, dims:]
    return torch.from_numpy(input_data).float(), torch.from_numpy(target_data).float()


# =====================================================
# Optimizer
# =====================================================

def get_optimizer(optimizer_name: str, learning_rate: float, model: torch.nn.Module):
    """Return a torch optimizer for the model."""
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    if name in ("adamw",):
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# =====================================================
# Training / Validation loops (used by 03_train_predictor.py)
# =====================================================

def _step_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_function: Callable,
) -> tuple[torch.Tensor, float]:
    """Compute loss for a batch, returning (loss_tensor, weight_for_averaging)."""
    outputs = model(inputs)
    requires_mask = bool(getattr(loss_function, "requires_mask", False))
    if requires_mask:
        mask = compute_non_missing_mask_from_inputs(inputs)
        loss = loss_function(outputs, targets, mask=mask)
        weight = float(mask.sum().item())
    else:
        loss = loss_function(outputs, targets)
        weight = float(inputs.size(0) * outputs.size(1))
    return loss, max(weight, 1.0)  # ensure non-zero


def train_model(
    model: torch.nn.Module,
    train_data_loader: DataLoader,
    val_data_loader: DataLoader,
    epochs: int,
    optimizer,
    loss_function: Callable,
    device: str = "cpu",
):
    """Train model for `epochs`, returning (train_loss_history, val_loss_history)."""
    train_hist: List[float] = []
    val_hist: List[float] = []

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running, weight_sum = 0.0, 0.0
        for inputs, targets in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, w = _step_loss(model, inputs, targets, loss_function)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * w
            weight_sum += w
        train_loss = running / max(weight_sum, 1e-8)
        train_hist.append(train_loss)

        # validation
        model.eval()
        running, weight_sum = 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_data_loader, desc=f"Epoch {epoch+1}/{epochs} [valid]"):
                inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
                targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
                loss, w = _step_loss(model, inputs, targets, loss_function)
                running += float(loss.item()) * w
                weight_sum += w
        val_loss = running / max(weight_sum, 1e-8)
        val_hist.append(val_loss)

    return train_hist, val_hist


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, data_loader: DataLoader, loss_function: Callable) -> float:
    """Evaluate average loss on a dataloader. Respects masked vs unmasked losses."""
    model.eval()
    running, weight_sum = 0.0, 0.0
    for inputs, targets in data_loader:
        inputs = inputs.to(device=next(model.parameters()).device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=inputs.device, dtype=torch.float32, non_blocking=True)
        loss, w = _step_loss(model, inputs, targets, loss_function)
        running += float(loss.item()) * w
        weight_sum += w
    return running / max(weight_sum, 1e-8)
