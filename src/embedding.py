"""
Learned ODD-similarity embedding.

Implements a SimCLR-style contrastive objective over H3 cells. The encoder
phi: R^F -> R^d (with d = 32) is a 3-layer MLP with ReLU activations and a
final L2-normalization layer. Positive pairs are cells whose HDV claim
frequency lies within a 10% tolerance band; negatives are all other cells
in the batch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ODDEncoder(nn.Module):
    """
    Three-layer MLP with ReLU activations and L2-normalized output, exactly
    as described in Section 5.2 of the paper.
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = F.normalize(z, p=2, dim=-1)
        return z


# ---------------------------------------------------------------------------
# Positive-pair sampling
# ---------------------------------------------------------------------------

class ContrastivePairDataset(Dataset):
    """
    Samples positive pairs for supervised contrastive learning over H3 cells.

    A positive pair is two cells from the **same city** whose HDV claim
    frequency also lies within a multiplicative tolerance band. Using city
    identity as the primary supervision signal matches the paper description
    (Section 5.2: supervised contrastive learning) and prevents cross-city
    pairs from collapsing the embedding. Within a city, the frequency band
    further filters to cells with similar local driving environments.

    """

    def __init__(
        self,
        features: np.ndarray,
        freq: np.ndarray,
        city_ids: np.ndarray,          # integer city label per cell
        tolerance: float = 0.10,
        rng_seed: int = 12345,
    ):
        self.features = features.astype(np.float32)
        self.freq = freq.astype(np.float32)
        self.city_ids = city_ids
        self.tolerance = float(tolerance)
        self.rng = np.random.default_rng(rng_seed)

        log_f = np.log(np.maximum(self.freq, 1e-9))
        bandwidth = np.log(1.0 + self.tolerance)

        self.partners: list[np.ndarray] = []
        for i in range(len(log_f)):
            city_i = city_ids[i]

            # Candidate positive: same city, similar frequency
            same_city = np.where(city_ids == city_i)[0]
            lo, hi = log_f[i] - bandwidth, log_f[i] + bandwidth
            cands = same_city[
                (log_f[same_city] >= lo) &
                (log_f[same_city] <= hi) &
                (same_city != i)
            ]

            if len(cands) == 0:
                # Relax: any other cell in the same city
                cands = same_city[same_city != i]

            if len(cands) == 0:
                # Last resort: nearest-frequency cell in ANY city (shouldn't
                # happen in practice since each city has ≥ 400 cells)
                order = np.argsort(np.abs(log_f - log_f[i]))
                cands = order[order != i][:5]

            self.partners.append(cands)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        j = int(self.rng.choice(self.partners[idx]))
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.features[j]),
        )


# ---------------------------------------------------------------------------
# NT-Xent loss (SimCLR objective)
# ---------------------------------------------------------------------------

def nt_xent_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Normalized temperature-scaled cross-entropy (NT-Xent) loss used in SimCLR.

    Given a batch of B anchors and B positives, build a 2B x 2B similarity
    matrix and treat each (anchor, positive) pair as one positive among
    2B - 1 negatives (excluding the diagonal).
    """
    batch_size = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)  # (2B, D)
    sim = z @ z.t()                   # cosine since z is normalized

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_masked = sim.masked_fill(mask, float("-inf"))

    # Positive index for row i:
    #   if i < B  -> i + B   (its augmented counterpart)
    #   if i >= B -> i - B
    pos_idx = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    logits = sim_masked / temperature
    loss = F.cross_entropy(logits, pos_idx)
    return loss


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    feature_columns: Sequence[str]
    embed_dim: int = 32
    hidden_dim: int = 64
    batch_size: int = 256
    epochs: int = 150
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    temperature: float = 0.10
    tolerance: float = 0.10
    seed: int = 20260525


def train_embedding(
    cell_features: pd.DataFrame,
    frequency_column: str,
    config: TrainConfig,
    out_dir: Path,
    verbose: bool = True,
) -> dict:
    """
    Train the encoder on the supplied cell features. Persists:
      - the encoder state_dict
      - the standardization parameters (mean, std) used at inference
      - the training config
      - the per-cell embedding vectors (cell_id -> 32-dim embedding)
      - the city-level mean embedding (city -> 32-dim embedding)
      - the resulting ODD-similarity matrix S over all cities
      - a JSON of training diagnostics (loss curve)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    feat_cols = list(config.feature_columns)
    X = cell_features[feat_cols].to_numpy(dtype=np.float32)
    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    X_std = (X - mu) / sd
    freq = cell_features[frequency_column].to_numpy(dtype=np.float32)

    # City integer labels — used as the primary supervision signal so that
    # positive pairs are within-city (supervised contrastive, Section 5.2).
    cities = cell_features["city"].tolist()
    city_vocab = {c: i for i, c in enumerate(sorted(set(cities)))}
    city_ids = np.array([city_vocab[c] for c in cities], dtype=np.int64)

    dataset = ContrastivePairDataset(
        X_std, freq, city_ids=city_ids,
        tolerance=config.tolerance, rng_seed=config.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    encoder = ODDEncoder(
        n_features=X.shape[1],
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
    )
    optim = torch.optim.Adam(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history = []
    for epoch in range(config.epochs):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_a, x_b in loader:
            z_a = encoder(x_a)
            z_b = encoder(x_b)
            loss = nt_xent_loss(z_a, z_b, temperature=config.temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        history.append(epoch_loss)
        if verbose and (epoch % 5 == 0 or epoch == config.epochs - 1):
            print(f"  epoch {epoch:3d}  loss={epoch_loss:.4f}")

    # Inference for all cells
    encoder.eval()
    with torch.no_grad():
        Z = encoder(torch.from_numpy(X_std)).cpu().numpy()

    # Persist
    torch.save(encoder.state_dict(), out_dir / "encoder.pt")
    np.save(out_dir / "cell_embeddings.npy", Z)
    np.save(out_dir / "feature_mean.npy", mu)
    np.save(out_dir / "feature_std.npy", sd)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2))
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))

    # City-level mean embedding (Section 5.3, area-weighted mean simplified to
    # uniform across cells since synthetic cells share unit area)
    df = cell_features[["cell_id", "city"]].copy()
    df_emb = pd.DataFrame(Z, columns=[f"e{j}" for j in range(Z.shape[1])])
    df = pd.concat([df.reset_index(drop=True), df_emb], axis=1)
    df.to_csv(out_dir / "cell_embeddings.csv", index=False)

    city_emb = df.drop(columns=["cell_id"]).groupby("city").mean()
    # Re-normalize to unit length so cosine-distance-based similarities work
    arr = city_emb.to_numpy()
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    city_emb_normed = pd.DataFrame(arr, index=city_emb.index, columns=city_emb.columns)
    city_emb_normed.to_csv(out_dir / "city_embeddings.csv")

    # ODD similarity matrix S_{c,c'} = exp( - ||phi(c) - phi(c')||^2 / (2 l^2) )
    # We fit the length-scale l so that the median pairwise similarity is ~ 0.5
    diffs = arr[:, None, :] - arr[None, :, :]
    dist2 = (diffs ** 2).sum(axis=-1)
    triu = dist2[np.triu_indices_from(dist2, k=1)]
    median_dist2 = float(np.median(triu))
    # Floor: ensure the length-scale is large enough that non-identical cities
    # don't all get similarity ≈ 1. Without a floor, a degenerate embedding
    # (small median_dist2) produces a tiny ell2, compressing all similarities
    # toward 1. The floor of 0.05 keeps the scale interpretable.
    ell2 = max(median_dist2 / (2.0 * np.log(2.0)), 0.05)
    S = np.exp(-dist2 / (2.0 * ell2))
    S_df = pd.DataFrame(S, index=city_emb.index, columns=city_emb.index)
    S_df.to_csv(out_dir / "city_similarity.csv")

    diagnostics = {
        "final_loss": history[-1],
        "median_pairwise_dist2": median_dist2,
        "length_scale_squared": float(ell2),
        "n_cells_trained": int(X.shape[0]),
        "embed_dim": config.embed_dim,
    }
    (out_dir / "embedding_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return diagnostics


def load_similarity(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(out_dir) / "city_similarity.csv", index_col=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--out", default="results/embedding", type=Path)
    parser.add_argument("--epochs", default=40, type=int)
    args = parser.parse_args()

    cells = pd.read_csv(args.data_dir / "cell_features.csv")

    # Section 5.2 feature set
    feature_cols = [
        "road_length_arterial_km",
        "road_length_collector_km",
        "road_length_local_km",
        "intersection_density_per_km2",
        "signalized_intersection_fraction",
        "betweenness_centrality_mean",
        "population_density_per_km2",
        "pedestrian_commute_share",
        "land_use_residential_share",
        "land_use_commercial_share",
        "historical_crash_density_per_mile",
    ]
    cfg = TrainConfig(feature_columns=feature_cols, epochs=args.epochs)
    diag = train_embedding(
        cell_features=cells,
        frequency_column="hdv_claim_freq_per_million_miles",
        config=cfg,
        out_dir=args.out,
    )
    print(json.dumps(diag, indent=2))
