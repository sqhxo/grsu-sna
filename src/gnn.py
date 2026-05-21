"""Graph neural networks for semi-supervised community classification.

GCN (Kipf & Welling, 2017) and GraphSAGE (Hamilton et al., 2017) with
stratified train/val/test splits, checkpoint save/load, and proper test-set
evaluation. Training can resume from a saved checkpoint to skip re-fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import igraph as ig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


def _build_features(graph: ig.Graph, feature_dim: int = 128) -> torch.Tensor:
    n = graph.vcount()
    if n <= feature_dim:
        base = torch.eye(n)
    else:
        edges = graph.get_edgelist()
        if not edges:
            base = torch.zeros(n, feature_dim - 2)
        else:
            src = np.array([u for u, v in edges] + [v for u, v in edges])
            dst = np.array([v for u, v in edges] + [u for u, v in edges])
            A = coo_matrix(
                (np.ones(len(src)), (src, dst)), shape=(n, n),
            ).tocsr()
            try:
                k = min(feature_dim - 2, n - 1)
                u, s, _vt = svds(A.astype(float), k=k)
                base = torch.tensor(u * s, dtype=torch.float32)
            except Exception:
                base = torch.randn(n, feature_dim - 2) * 0.01

    deg = np.array(graph.degree(), dtype=np.float32)
    log_deg = np.log1p(deg).reshape(-1, 1)
    lcc = np.array(graph.transitivity_local_undirected(mode="zero"),
                   dtype=np.float32).reshape(-1, 1)
    aux = torch.tensor(np.hstack([log_deg, lcc]), dtype=torch.float32)
    return torch.cat([base, aux], dim=1)


def _build_normalized_adj(graph: ig.Graph) -> torch.Tensor:
    """Symmetric normalized adjacency with self-loops: D^-1/2 (A + I) D^-1/2."""
    n = graph.vcount()
    edges = graph.get_edgelist()
    src = [u for u, v in edges] + [v for u, v in edges] + list(range(n))
    dst = [v for u, v in edges] + [u for u, v in edges] + list(range(n))
    data = np.ones(len(src), dtype=np.float32)
    A = coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    coo = A.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(
        coo.data * deg_inv_sqrt[coo.row] * deg_inv_sqrt[coo.col],
        dtype=torch.float32,
    )
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def _build_edge_index(graph: ig.Graph) -> torch.Tensor:
    edges = graph.get_edgelist()
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    return torch.tensor([src, dst], dtype=torch.long)


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(adj, x)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.sparse.mm(adj, h)
        return self.lin2(h)


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE with mean aggregation."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.5):
        super().__init__()
        self.W1_self = nn.Linear(in_dim, hidden_dim)
        self.W1_neigh = nn.Linear(in_dim, hidden_dim)
        self.W2_self = nn.Linear(hidden_dim, out_dim)
        self.W2_neigh = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    @staticmethod
    def _mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device)
        deg.index_add_(0, dst, torch.ones_like(src, dtype=torch.float32))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        return out / deg

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        neigh = self._mean_aggregate(x, edge_index)
        h = F.relu(self.W1_self(x) + self.W1_neigh(neigh))
        h = F.dropout(h, p=self.dropout, training=self.training)
        neigh = self._mean_aggregate(h, edge_index)
        return self.W2_self(h) + self.W2_neigh(neigh)


@dataclass
class TrainResult:
    labels_pred: np.ndarray
    history: dict[str, list[float]]
    train_acc: float
    val_acc: float
    test_acc: float
    model_name: str
    train_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    val_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    test_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    epochs_trained: int = 0
    loaded_from_cache: bool = False


def _train_val_test_split(labels: np.ndarray,
                          train_ratio: float = 0.6,
                          val_ratio: float = 0.2,
                          seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split: at least one sample per class in train and (when
    possible) one in val and one in test. Remaining ratio goes to test."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_class = len(idx)
        if n_class == 1:
            train_mask[idx[0]] = True
            continue
        if n_class == 2:
            train_mask[idx[0]] = True
            test_mask[idx[1]] = True
            continue
        n_train = max(1, int(round(n_class * train_ratio)))
        n_val = max(1, int(round(n_class * val_ratio)))
        n_train = min(n_train, n_class - 2)
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask


def _build_model(model_name: str, in_dim: int, hidden_dim: int,
                 out_dim: int, dropout: float = 0.5) -> nn.Module:
    if model_name == "gcn":
        return GCN(in_dim, hidden_dim, out_dim, dropout=dropout)
    if model_name == "graphsage":
        return GraphSAGE(in_dim, hidden_dim, out_dim, dropout=dropout)
    raise ValueError(f"Unknown model: {model_name}")


def _forward_args(graph: ig.Graph, model_name: str, x: torch.Tensor):
    if model_name == "gcn":
        return (x, _build_normalized_adj(graph))
    if model_name == "graphsage":
        return (x, _build_edge_index(graph))
    raise ValueError(model_name)


def _evaluate(model, forward_args, labels, mask) -> float:
    if not mask.any():
        return float("nan")
    model.eval()
    with torch.no_grad():
        pred = model(*forward_args).argmax(dim=1).cpu().numpy()
    return float((pred[mask] == labels[mask]).mean())


def save_checkpoint(path: str | Path, model: nn.Module, result: TrainResult,
                    config: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": config,
        "train_mask": result.train_mask,
        "val_mask": result.val_mask,
        "test_mask": result.test_mask,
        "history": result.history,
        "train_acc": result.train_acc,
        "val_acc": result.val_acc,
        "test_acc": result.test_acc,
        "epochs_trained": result.epochs_trained,
        "model_name": result.model_name,
    }, path)


def load_checkpoint(path: str | Path, graph: ig.Graph) -> tuple[nn.Module, dict]:
    """Restore a model + metadata for inference."""
    data = torch.load(path, weights_only=False, map_location="cpu")
    cfg = data["config"]
    model = _build_model(cfg["model_name"], cfg["in_dim"],
                         cfg["hidden_dim"], cfg["out_dim"],
                         dropout=cfg.get("dropout", 0.5))
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model, data


def train_gnn(
    graph: ig.Graph,
    labels: np.ndarray,
    model_name: str = "gcn",
    hidden_dim: int = 64,
    epochs: int = 500,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    patience: int = 50,
    seed: int = 42,
    checkpoint_path: str | Path | None = None,
    force_retrain: bool = False,
    verbose: bool = False,
) -> TrainResult:
    """Train a GCN/GraphSAGE. If `checkpoint_path` exists, reuse it unless
    `force_retrain=True`. Saves the model and the train/val/test masks so
    test-set evaluation is reproducible across runs."""

    if checkpoint_path is not None and not force_retrain and Path(checkpoint_path).exists():
        model, data = load_checkpoint(checkpoint_path, graph)
        x = _build_features(graph, feature_dim=data["config"]["in_dim"])
        fwd = _forward_args(graph, model_name, x)
        with torch.no_grad():
            pred_final = model(*fwd).argmax(dim=1).cpu().numpy()
        return TrainResult(
            labels_pred=pred_final,
            history=data["history"],
            train_acc=data["train_acc"],
            val_acc=data["val_acc"],
            test_acc=data["test_acc"],
            model_name=data["model_name"],
            train_mask=data["train_mask"],
            val_mask=data["val_mask"],
            test_mask=data["test_mask"],
            epochs_trained=data["epochs_trained"],
            loaded_from_cache=True,
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_classes = int(labels.max() + 1)
    x = _build_features(graph)
    in_dim = x.size(1)

    train_mask, val_mask, test_mask = _train_val_test_split(
        labels, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed,
    )
    y = torch.tensor(labels, dtype=torch.long)
    train_idx = torch.tensor(np.where(train_mask)[0], dtype=torch.long)
    val_idx = torch.tensor(np.where(val_mask)[0], dtype=torch.long)
    test_idx = torch.tensor(np.where(test_mask)[0], dtype=torch.long)

    model = _build_model(model_name, in_dim, hidden_dim, n_classes, dropout=dropout)
    fwd = _forward_args(graph, model_name, x)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state: dict | None = None
    no_improve = 0
    epochs_trained = 0
    history: dict[str, list[float]] = {"loss": [], "train_acc": [],
                                        "val_acc": [], "test_acc": []}

    for epoch in range(epochs):
        model.train()
        optim.zero_grad()
        logits = model(*fwd)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optim.step()
        epochs_trained = epoch + 1

        model.eval()
        with torch.no_grad():
            logits = model(*fwd)
            pred = logits.argmax(dim=1)
            train_acc = float((pred[train_idx] == y[train_idx]).float().mean())
            val_acc = (float((pred[val_idx] == y[val_idx]).float().mean())
                       if len(val_idx) else float("nan"))
            test_acc = (float((pred[test_idx] == y[test_idx]).float().mean())
                        if len(test_idx) else float("nan"))

        history["loss"].append(float(loss.item()))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        if verbose and epoch % 25 == 0:
            print(f"  epoch {epoch:3d} | loss={loss.item():.4f} "
                  f"train={train_acc:.3f} val={val_acc:.3f} test={test_acc:.3f}")

        # Track best by val (fall back to train when val is empty).
        score = val_acc if not np.isnan(val_acc) else train_acc
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_acc_final = _evaluate(model, fwd, labels, train_mask)
    val_acc_final = _evaluate(model, fwd, labels, val_mask)
    test_acc_final = _evaluate(model, fwd, labels, test_mask)

    model.eval()
    with torch.no_grad():
        pred_final = model(*fwd).argmax(dim=1).cpu().numpy()

    result = TrainResult(
        labels_pred=pred_final,
        history=history,
        train_acc=train_acc_final,
        val_acc=val_acc_final,
        test_acc=test_acc_final,
        model_name=model_name,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        epochs_trained=epochs_trained,
    )

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, result, config={
            "model_name": model_name,
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "out_dim": n_classes,
            "dropout": dropout,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        })

    return result
