"""Neural Granger causality baseline: component-wise MLP (cMLP).

Implements the cMLP model of Tank et al., "Neural Granger Causality for
Nonlinear Time Series" (IEEE TPAMI 44(8), 2022; arXiv:1802.05842), following
the reference implementation github.com/iancovert/Neural-GC (MIT license):
one MLP per target series maps all series' lagged values ``X(t-1..t-L)``
(flattened, lag-major, matching :func:`stcg.make_lagged_design`) to the
target's next value under an MSE loss, while a group-lasso (L2,1) penalty
over the first-layer input columns grouped by ``(source series, lag)``
induces structured sparsity. Post-training group norms give a
``(lag, source)`` importance matrix per target, assembled into the
repository score convention ``(max_lag, n_nodes, n_nodes)`` with self-edges
zeroed, so :func:`stcg.recovery_summary` consumes it like any other
baseline.

Training uses full-batch Adam on the smooth MSE term followed by an exact
proximal group soft-thresholding step on the first-layer columns (the same
update as Neural-GC's ``prox_update``), which is deterministic given the
per-``(seed, target)`` ``torch.manual_seed``. Inputs follow the same
leakage-free-by-construction unsupervised convention as the other wrappers
(``pcmci_parcorr_scores``, ``msvar_scores``): the full observed series is
z-scored, then the lagged design and targets are z-scored again, exactly
like :func:`stcg.ridge_var_scores`.
"""

from __future__ import annotations

import numpy as np

from .baselines import _zscore, make_lagged_design

ACTIVATIONS = ("tanh", "relu")


def _fit_cmlp_target(
    design: np.ndarray,
    target: np.ndarray,
    n_nodes: int,
    max_lag: int,
    hidden: int,
    penalty: float,
    lr: float,
    epochs: int,
    activation: str,
    seed: int,
    optimizer_name: str,
    momentum: float,
    weight_decay: float,
) -> tuple[np.ndarray, float]:
    """Fit one cMLP target equation and return its first-layer group norms.

    Returns ``(norms, final_loss)`` where ``norms`` has shape
    ``(max_lag, n_nodes)`` — the L2 norm of each first-layer input column,
    i.e. the importance of ``(lag, source)`` for this target. Single-threaded
    CPU execution keeps the fit deterministic and avoids oversubscribing
    cores when callers loop over targets.
    """

    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(seed)
        n_features = design.shape[1]
        activation_cls = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[activation]
        model = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            activation_cls(),
            torch.nn.Linear(hidden, 1),
        )
        inputs = torch.as_tensor(design, dtype=torch.float32)
        labels = torch.as_tensor(target, dtype=torch.float32).unsqueeze(1)
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            # ISTA-style SGD (+ optional momentum) with a proximal step, as
            # in Neural-GC's train_model_ista: unlike Adam, SGD does not
            # renormalize per-parameter gradients, so the uniform group
            # shrinkage below balances each column's predictive gradient
            # and weak (noise) columns are driven exactly to zero.
            optimizer = torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        mse = torch.nn.MSELoss()

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = mse(model(inputs), labels)
            loss.backward()
            optimizer.step()
            # Proximal group-lasso step (Neural-GC prox_update): each input
            # column is its own (source, lag) group, so shrink columns by
            # max(0, 1 - lr * penalty / ||column||_2).
            with torch.no_grad():
                weight = model[0].weight  # (hidden, n_features)
                norms = weight.norm(dim=0).clamp_min(1e-12)
                shrink = torch.clamp(1.0 - lr * penalty / norms, min=0.0)
                weight.mul_(shrink.unsqueeze(0))

        with torch.no_grad():
            final_loss = float(mse(model(inputs), labels))
            column_norms = model[0].weight.norm(dim=0).cpu().numpy().astype(float)
    finally:
        torch.set_num_threads(previous_threads)

    return column_norms.reshape(max_lag, n_nodes), final_loss


def cmlp_scores(
    x: np.ndarray,
    max_lag: int,
    hidden: int = 8,
    penalty: float = 0.15,
    lr: float = 0.05,
    epochs: int = 300,
    activation: str = "tanh",
    seed: int = 0,
    optimizer: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Score lagged edges with a component-wise MLP neural-Granger model.

    For each target node, a one-hidden-layer MLP predicts ``x_j(t)`` from all
    nodes' lags with a group-lasso penalty on the first-layer input columns
    grouped by ``(source, lag)``; the post-training column L2 norms score
    each edge. Self-edges are zeroed and scores are non-negative and finite,
    matching the ``(lag, source, target)`` convention of every other wrapper.

    The default budget (``hidden=8``, ``penalty=0.15``, ``lr=0.05``,
    ``epochs=300``, ``momentum=0.9``, tanh, ISTA-style SGD+prox) was fixed by
    a pilot sweep on seeds 0-4 of the repository's three synthetic
    generators: SGD is essential (Adam's per-parameter gradient rescaling
    breaks the shrinkage equilibrium and inverts the score ranking on
    low-SNR linear VARs), and a small hidden layer regularizes the
    low signal-to-noise linear cases. On that pilot the defaults reach
    lagged-edge AUC ~0.92 (var), ~0.97 (nonlinear_var), ~0.80
    (switching_var) at roughly 0.6 s per series set (8 targets) on CPU.

    Determinism: target ``j`` of dataset ``seed`` fits with
    ``torch.manual_seed(seed * 1000 + j)``, single CPU thread, so repeated
    calls return identical scores.

    When ``return_info`` is true, a diagnostics dict is returned alongside
    the scores with the per-target final training MSE and the fixed training
    budget (``hidden``, ``penalty``, ``lr``, ``epochs``, ``activation``), so
    runners can record the hyperparameters of every row.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected x with shape (timesteps, n_nodes), got {x.shape}")
    timesteps, n_nodes = x.shape
    if timesteps <= max_lag + 1:
        raise ValueError("Need more timesteps than max_lag + 1")
    if hidden < 1:
        raise ValueError("hidden must be at least 1")
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation must be one of {ACTIVATIONS}")
    if optimizer not in ("sgd", "adam"):
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    z = _zscore(x)
    design, target = make_lagged_design(z, max_lag)
    design = _zscore(design)
    target = _zscore(target)

    scores = np.zeros((max_lag, n_nodes, n_nodes), dtype=float)
    final_losses: list[float] = []
    for target_idx in range(n_nodes):
        norms, final_loss = _fit_cmlp_target(
            design,
            target[:, target_idx],
            n_nodes=n_nodes,
            max_lag=max_lag,
            hidden=hidden,
            penalty=penalty,
            lr=lr,
            epochs=epochs,
            activation=activation,
            seed=seed * 1000 + target_idx,
            optimizer_name=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        scores[:, :, target_idx] = norms
        final_losses.append(final_loss)

    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    for node in range(n_nodes):
        scores[:, node, node] = 0.0

    if not return_info:
        return scores

    info: dict[str, object] = {
        "final_losses": final_losses,
        "hidden": hidden,
        "penalty": penalty,
        "lr": lr,
        "epochs": epochs,
        "activation": activation,
        "optimizer": optimizer,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "n_samples": int(design.shape[0]),
    }
    return scores, info
