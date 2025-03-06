import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from tqdm import tqdm


def naive_loss(n_features: int, d_mlp: int, p: float) -> float:
    """Naive loss from monosemantic ReLUs and orthogonal(!) embeddings."""
    return (n_features - d_mlp) / n_features * p / 6


@dataclass
class Config:
    """Configuration for the ResidualMLP model and training."""

    n_features: int = 100
    d_embed: int = 1000
    d_mlp: int = 50
    embed: Literal["random", "identity"] = "random"
    seed: int | None = None
    feature_probability: float = 0.01
    batch_size: int = 2048
    steps: int = 10_000
    lr: float = 3e-3
    print_freq: int = 500
    device: str = "cuda"


class MLP(nn.Module):
    """A simple MLP module."""

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp

        self.mlp_in: Float[Tensor, "d_model d_mlp"] = nn.Parameter(torch.empty(d_model, d_mlp))
        self.mlp_out: Float[Tensor, "d_mlp d_model"] = nn.Parameter(torch.empty(d_mlp, d_model))

        nn.init.kaiming_uniform_(self.mlp_in)
        nn.init.kaiming_uniform_(self.mlp_out)

    def forward(self, x: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch d_model"]:
        mid_pre_act: Float[Tensor, "batch d_mlp"] = einops.einsum(
            x, self.mlp_in, "batch d_model, d_model d_mlp -> batch d_mlp"
        )
        mid: Float[Tensor, "batch d_mlp"] = F.relu(mid_pre_act)
        out: Float[Tensor, "batch d_model"] = einops.einsum(
            mid, self.mlp_out, "batch d_mlp, d_mlp d_model -> batch d_model"
        )
        return out


class ResidualMLPModel(nn.Module):
    """A simple residual MLP model with one layer."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        if config.embed == "random":
            W_E: Float[Tensor, "n_features d_embed"] = torch.randn(
                config.n_features, config.d_embed, device=config.device
            )
        elif config.embed == "identity":
            W_E: Float[Tensor, "n_features d_embed"] = torch.eye(
                config.n_features, config.d_embed, device=config.device
            )
        else:
            raise ValueError(f"Unknown embedding type {config.embed}")
        W_E = F.normalize(W_E, dim=1)
        self.register_buffer("W_E", W_E)

        self.mlp = MLP(
            d_model=config.d_embed,
            d_mlp=config.d_mlp,
        )

    def forward(self, x: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        residual = einops.einsum(
            x, self.W_E, "batch n_features, n_features d_embed -> batch d_embed"
        )
        mlp_out: Float[Tensor, "batch d_embed"] = self.mlp(residual)
        residual = residual + mlp_out
        out: Float[Tensor, "batch n_features"] = einops.einsum(
            residual, self.W_E, "batch d_embed, n_features d_embed -> batch n_features"
        )
        return out


class SparseFeatureDataset:
    """Dataset that generates sparse feature vectors."""

    def __init__(
        self,
        config: Config,
    ):
        self.n_features = config.n_features
        self.feature_probability = config.feature_probability
        self.device = config.device

    def generate_batch(
        self, batch_size: int
    ) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch n_features"]]:
        batch = torch.zeros((batch_size, self.n_features), device=self.device)
        mask = torch.rand_like(batch) < self.feature_probability
        values = torch.rand_like(batch) * 2 - 1
        batch = values * mask
        labels = F.relu(batch) + batch
        return batch, labels


def train(config: Config) -> ResidualMLPModel:
    # Generate a random number
    seed = config.seed or int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ResidualMLPModel(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0)

    dataset = SparseFeatureDataset(config)

    pbar = tqdm(range(config.steps), desc="Training")
    for step in pbar:
        batch, labels = dataset.generate_batch(config.batch_size)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = ((outputs - labels) ** 2).mean(dim=(0, 1))
        loss.backward()
        optimizer.step()

        if step % config.print_freq == 0 or step == config.steps - 1:
            pbar.set_postfix({"loss": f"{loss.item():.2e}"})

    return model


def evaluate(
    model: ResidualMLPModel, dataset: SparseFeatureDataset, batch_size: int = 10_000
) -> float:
    with torch.no_grad():
        batch, labels = dataset.generate_batch(batch_size)
        outputs = model(batch)
        loss = ((outputs - labels) ** 2).mean(dim=(0, 1)).item()
    return loss


def plot_loss_of_input_sparsity(
    models: ResidualMLPModel | list[ResidualMLPModel],
    feature_probabilities: Iterable[float],
    config: Config,
    batch_size: int = 100_000,
    labels: list[str] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots()
    models = [models] if not isinstance(models, list) else models

    dataset = SparseFeatureDataset(config)
    feature_probabilities = np.array(feature_probabilities)
    naive_losses = naive_loss(config.n_features, config.d_mlp, feature_probabilities)

    for i, model in enumerate(models):
        with torch.no_grad():
            losses = []
            for feature_probability in feature_probabilities:
                dataset.feature_probability = feature_probability
                loss = evaluate(model, dataset, batch_size=batch_size)
                losses.append(loss)
        losses = np.array(losses)

        ax.scatter(
            feature_probabilities,
            losses / feature_probabilities,
            s=1,
            label=labels[i] if labels else None,
        )

    ax.plot(
        feature_probabilities,
        naive_losses / feature_probabilities,
        label="Naive loss",
        color="k",
        ls="--",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Feature probability p = (1-S)")
    ax.set_ylabel("Adjusted loss L / (1-S)")
    ax.legend(ncols=3, loc="lower center")
    return fig


def plot_outputs(model: ResidualMLPModel, config: Config) -> plt.Figure:
    fig, ax = plt.subplots()
    dataset = SparseFeatureDataset(config)
    batch, labels = dataset.generate_batch(1)
    with torch.no_grad():
        outputs = model(batch)
    ax.scatter(range(config.n_features), outputs[0].cpu(), label="outputs", s=1)
    ax.scatter(range(config.n_features), labels[0].cpu(), label="labels", s=1)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Output value")
    active_features = torch.where(batch[0] != 0)[0]
    ax.set_title(f"Outputs for batch with features {active_features.tolist()}")
    ax.legend()
    return fig


def plot_2_feature_cases(model: ResidualMLPModel, config: Config) -> plt.Figure:
    fig, ax = plt.subplots()
    batch = torch.zeros((100, config.n_features), device=config.device)
    batch[:, 0] = torch.linspace(-1, 1, 100)
    with torch.no_grad():
        outputs = model(batch)
    ax.plot(
        batch[:, 0].cpu(),
        outputs[:, 0].cpu(),
        label=f"feature 0, when feature 1 active",
        color="C0",
        ls="-",
    )
    ax.plot(
        batch[:, 0].cpu(),
        outputs[:, 1].cpu(),
        label=f"feature 0, when feature 1 inactive",
        color="C0",
        ls=":",
    )

    batch[:, 0] = 0
    batch[:, 1] = torch.linspace(-1, 1, 100)
    with torch.no_grad():
        outputs = model(batch)
    ax.plot(
        batch[:, 1].cpu(),
        outputs[:, 0].cpu(),
        label=f"feature 1, when feature 0 active",
        color="C1",
        ls="-",
    )
    ax.plot(
        batch[:, 1].cpu(),
        outputs[:, 1].cpu(),
        label=f"feature 1, when feature 0 inactive",
        color="C1",
        ls=":",
    )
    ax.legend()
    return fig


if __name__ == "__main__":
    config = Config(
        n_features=2,
        d_embed=2,
        d_mlp=1,
        feature_probability=0.1,
        embed="identity",
        steps=5_000,
        batch_size=2048,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    models = []
    for _ in range(50):
        model = train(config)
        models.append(model)

    fig = plot_outputs(models[-1], config)
    fig.show()
    fig = plot_2_feature_cases(models[-1], config)
    fig.show()
    print("Naive loss:", naive_loss(config.n_features, config.d_mlp, config.feature_probability))

    fig, [[ax1, ax2], [ax3, ax4], [ax5, ax6]] = plt.subplots(3, 2, figsize=(15, 15))
    dataset = SparseFeatureDataset(config)

    losses = []
    for model in models:
        loss = evaluate(model, dataset, batch_size=10_000)
        losses.append(loss)

    norm = plt.Normalize(vmin=min(losses), vmax=max(losses))

    for i, model in enumerate(models):
        print(f"Model {i} weights:")
        print("W_in:", model.mlp.mlp_in.cpu().detach().numpy().flatten())
        print("W_out:", model.mlp.mlp_out.cpu().detach().numpy().flatten())
        loss = losses[i]
        print("Loss:", loss)
        color = plt.cm.viridis(norm(loss))

        # Plot scatter of W_in
        Win = model.mlp.mlp_in.cpu().detach().numpy().flatten()
        ax1.scatter(Win[0], Win[1], s=10, color=color)
        ax1.set_xlabel("W_in feature 0")
        ax1.set_ylabel("W_in feature 1")

        Wout = model.mlp.mlp_out.cpu().detach().numpy().flatten()
        ax2.scatter(Wout[0], Wout[1], s=10, color=color)
        ax2.set_xlabel("W_out feature 0")
        ax2.set_ylabel("W_out feature 1")

        # Product of W_in and W_out
        ax3.scatter(Win[0] * Wout[0], Win[1] * Wout[1], s=10, color=color)
        ax3.set_xlabel("W_in feature 0 * W_out feature 0")
        ax3.set_ylabel("W_in feature 1 * W_out feature 1")

        ax4.scatter(Win[0] * Wout[1], Win[1] * Wout[0], s=10, color=color)
        ax4.set_xlabel("W_in feature 0 * W_out feature 1")
        ax4.set_ylabel("W_in feature 1 * W_out feature 0")

        ax5.scatter(Win[0] * Wout[0], Win[0] * Wout[1], s=10, color=color)
        ax5.set_xlabel("W_in feature 0 * W_out feature 0")
        ax5.set_ylabel("W_in feature 0 * W_out feature 1")

        ax6.scatter(Win[0] * Wout[0], Win[1] * Wout[0], s=10, color=color)
        ax6.set_xlabel("W_in feature 0 * W_out feature 0")
        ax6.set_ylabel("W_in feature 1 * W_out feature 0")

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    fig.colorbar(sm, ax=[ax1, ax2, ax3, ax4, ax5, ax6], label="Loss")
    fig.show()
