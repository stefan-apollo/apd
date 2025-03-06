from dataclasses import dataclass

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm


def naive_loss(n_features: int, d_mlp: int, p: float, embed: str) -> float:
    if embed == "random" or embed == "identity":
        # d_mlp features perfectly (loss 0), the others not at all (loss 1/6
        # because 0.5*\int_0^1 x^2 = 1/3 if active) so (n_features - d_mlp) * 1/6 * p
        loss = (n_features - d_mlp) * p / 6
    else:
        raise ValueError(f"Unknown embedding type {embed}")
    return loss / n_features


@dataclass
class Config:
    """Configuration for the ResidualMLP model and training."""

    # Model parameters
    n_features: int = 100  # Number of input features
    d_embed: int = 100  # Embedding dimension
    d_mlp: int = 50  # Hidden layer size
    n_instances: int = 1  # Number of model instances

    # Training parameters
    seed: int = 0
    feature_probability: float = 0.01  # Probability of a feature being active
    batch_size: int = 256
    steps: int = 5000
    lr: float = 3e-3
    print_freq: int = 500


class MLP(nn.Module):
    """A simple MLP module."""

    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        n_instances: int = 1,
    ):
        super().__init__()
        self.n_instances = n_instances
        self.d_model = d_model
        self.d_mlp = d_mlp

        # Initialize weights
        self.mlp_in = nn.Parameter(torch.empty(n_instances, d_model, d_mlp))
        self.mlp_out = nn.Parameter(torch.empty(n_instances, d_mlp, d_model))

        # Initialize parameters with Kaiming initialization
        nn.init.kaiming_uniform_(self.mlp_in)
        nn.init.kaiming_uniform_(self.mlp_out)

        # No bias in this implementation for simplicity

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape [batch, n_instances, d_model]

        Returns:
            Output tensor of shape [batch, n_instances, d_model]
        """
        mid_pre_act = einops.einsum(
            x,
            self.mlp_in,
            "batch n_instances d_model, n_instances d_model d_mlp -> batch n_instances d_mlp",
        )
        mid = F.relu(mid_pre_act)
        out = einops.einsum(
            mid,
            self.mlp_out,
            "batch n_instances d_mlp, n_instances d_mlp d_model -> batch n_instances d_model",
        )
        return out


class ResidualMLPModel(nn.Module):
    """A simple residual MLP model with one layer."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Embedding matrices
        self.W_E = nn.Parameter(torch.empty(config.n_instances, config.n_features, config.d_embed))
        self.W_U = nn.Parameter(torch.empty(config.n_instances, config.d_embed, config.n_features))

        # Initialize embedding matrices
        nn.init.normal_(self.W_E, std=0.02)
        nn.init.normal_(self.W_U, std=0.02)

        # Create the MLP layer
        self.mlp = MLP(
            d_model=config.d_embed,
            d_mlp=config.d_mlp,
            n_instances=config.n_instances,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the ResidualMLP model.

        Args:
            x: Input tensor of shape [batch, n_instances, n_features]

        Returns:
            Output tensor of shape [batch, n_instances, n_features]
        """
        # Project input to embedding space
        residual = einops.einsum(
            x,
            self.W_E,
            "batch n_instances n_features, n_instances n_features d_embed -> batch n_instances d_embed",
        )

        # Apply MLP with residual connection
        out = self.mlp(residual)
        residual = residual + out

        # Project back to feature space
        out = einops.einsum(
            residual,
            self.W_U,
            "batch n_instances d_embed, n_instances d_embed n_features -> batch n_instances n_features",
        )

        return out


class SparseFeatureDataset:
    """Dataset that generates sparse feature vectors."""

    def __init__(
        self,
        n_instances: int,
        n_features: int,
        feature_probability: float,
        device: str,
    ):
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_probability = feature_probability
        self.device = device

    def generate_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """
        Generate a batch of sparse input features and corresponding labels.

        Args:
            batch_size: Number of samples in the batch

        Returns:
            Tuple of (inputs, labels)
            - inputs: Tensor of shape [batch_size, n_instances, n_features]
            - labels: Tensor of shape [batch_size, n_instances, n_features]
        """
        # Generate input batch where each feature has a probability of being non-zero
        batch = torch.zeros(batch_size, self.n_instances, self.n_features, device=self.device)
        mask = torch.rand_like(batch) < self.feature_probability

        # Set non-zero values to random values between [-1, 1]
        values = torch.rand_like(batch) * 2 - 1
        batch = values * mask

        # Calculate labels using act_fn(x) + x
        labels = F.relu(batch) + batch

        return batch, labels


def train_and_evaluate(config: Config, device: str = "cpu"):
    """Train and evaluate the ResidualMLP model."""
    # Set random seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create model
    model = ResidualMLPModel(config).to(device)

    # Create dataset and optimizer
    dataset = SparseFeatureDataset(
        n_instances=config.n_instances,
        n_features=config.n_features,
        feature_probability=config.feature_probability,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    # Training loop
    pbar = tqdm(range(config.steps), desc="Training")
    for step in pbar:
        # Generate a batch
        batch, labels = dataset.generate_batch(config.batch_size)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)

        # Compute loss (MSE)
        loss = ((outputs - labels) ** 2).mean()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Log progress
        if step % config.print_freq == 0 or step == config.steps - 1:
            pbar.set_postfix({"loss": f"{loss.item():.2e}"})

    # Evaluation
    print("\nEvaluation:")
    model.eval()
    with torch.no_grad():
        # Generate a test batch
        test_batch, test_labels = dataset.generate_batch(1000)

        # Forward pass
        test_outputs = model(test_batch)

        # Compute MSE loss
        test_loss = ((test_outputs - test_labels) ** 2).mean().item()

        # Compute fraction of positive activations to check sparsity
        active_inputs = (test_batch != 0).float().mean().item()

        print(f"Test MSE: {test_loss:.4e}")
        nl = naive_loss(
            config.n_features, config.d_mlp, config.feature_probability, embed="random"
        )
        print(f"Naive loss: {nl:.4e}")
        print(f"Input sparsity: {active_inputs:.4f} (target: {config.feature_probability:.4f})")

        # Create an input that is 0 everywhere except for a single feature 42
        test_batch = torch.zeros(1, config.n_instances, config.n_features, device=device)
        test_batch[:, :, 42] = 1
        test_labels = F.relu(test_batch) + test_batch
        test_labels = test_labels.to("cpu").detach().numpy()
        test_outputs = model(test_batch).to("cpu").detach().numpy()
        # Scatter outputs and labels
        plt.scatter(range(config.n_features), test_outputs[0, 0], label="outputs", s=1)
        # plt.scatter(range(config.n_features), test_labels[0, 0], label="labels")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create config
    config = Config(
        n_features=100,
        d_embed=100,
        d_mlp=50,
        n_instances=1,
        feature_probability=0.01,
        steps=5000,
        batch_size=256,
    )

    # Train and evaluate model
    model = train_and_evaluate(config, device)
