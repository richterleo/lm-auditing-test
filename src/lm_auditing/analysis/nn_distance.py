import importlib
import torch
import sys
from pathlib import Path

# # Add paths to sys.path if not already present
# project_root = Path(__file__).resolve().parents[2]
# if str(project_root) not in sys.path:
#     sys.path.append(str(project_root))

# orig_models = importlib.import_module("deep-anytime-testing.models.mlp", package="deep-anytime-testing")
# MLP = getattr(orig_models, "MLP")

from lm_auditing.utils.dat_wrapper import MLP


class CMLP(MLP):
    """
    CMLP  is an extension of the base MLP (Multi-Layer Perceptron).

    This class requires a configuration object `cfg` to specify the neural network parameters.
    The forward method implements a custom operation over the outputs of the base MLP.
    """

    def __init__(
        self,
        input_size,
        hidden_layer_size,
        output_size,
        layer_norm,
        drop_out,
        drop_out_p,
        bias,
        flatten=True,
    ):
        # Initialize the base MLP
        super(CMLP, self).__init__(
            input_size,
            hidden_layer_size,
            output_size,
            layer_norm,
            drop_out,
            drop_out_p,
            bias,
        )
        self.sigma = torch.nn.Tanh()
        self.flatten = flatten

    def forward(self, x, y) -> torch.Tensor:
        if len(x.shape) > 2 or len(y.shape) > 2:
            if self.flatten:
                x = torch.flatten(x, start_dim=1)
                y = torch.flatten(y, start_dim=1)
                g_x = self.model(x)
                g_y = self.model(y)
            else:
                num_samples = x.shape[-1]
                g_x, g_y = 0, 0
                for i in range(num_samples):
                    g_x += self.model(torch.flatten(x[..., i], start_dim=1)) / num_samples
                    g_y += self.model(torch.flatten(y[..., i], start_dim=1)) / num_samples
        else:
            g_x = self.model(x)
            g_y = self.model(y)

        output = torch.log(1 + 0.5 * self.sigma(g_x) - 0.5 * self.sigma(g_y))
        return output


class OptCMLP(MLP):
    """
    Optimized version of CMLP with batch processing and GPU support.
    Implements efficient vectorized operations and memory management.
    """

    def __init__(
        self,
        input_size,
        hidden_layer_size,
        output_size,
        layer_norm,
        drop_out,
        drop_out_p,
        bias,
        flatten=True,
        batch_size=32,
    ):
        super(OptCMLP, self).__init__(
            input_size,
            hidden_layer_size,
            output_size,
            layer_norm,
            drop_out,
            drop_out_p,
            bias,
        )
        self.sigma = torch.nn.Tanh()
        self.flatten = flatten
        self.batch_size = batch_size

    @torch.no_grad()
    def process_batch(self, x_batch):
        """Process a batch of inputs efficiently"""
        if self.flatten:
            x_batch = torch.flatten(x_batch, start_dim=1)
        return self.model(x_batch)

    def forward(self, x, y) -> torch.Tensor:
        """Optimized forward pass with batch processing"""
        if len(x.shape) > 2 or len(y.shape) > 2:
            if self.flatten:
                # Vectorized flattening
                x = torch.flatten(x, start_dim=1)
                y = torch.flatten(y, start_dim=1)
                g_x = self.model(x)
                g_y = self.model(y)
            else:
                # Efficient batch processing for multi-dimensional inputs
                x_reshaped = x.reshape(-1, *x.shape[1:-1], x.shape[-1])
                y_reshaped = y.reshape(-1, *y.shape[1:-1], y.shape[-1])

                # Vectorized operations instead of loop
                g_x = torch.mean(self.model(torch.flatten(x_reshaped, start_dim=1)), dim=0)
                g_y = torch.mean(self.model(torch.flatten(y_reshaped, start_dim=1)), dim=0)
        else:
            # Standard processing for 2D tensors
            g_x = self.model(x)
            g_y = self.model(y)

        # Optimized betting score computation
        score = torch.log1p(0.5 * (self.sigma(g_x) - self.sigma(g_y)))
        return score

    def predict_large_input(self, x, y, use_gpu=False):
        """Method for handling very large inputs with batch processing"""
        if use_gpu and torch.cuda.is_available():
            self.cuda()
            x = x.cuda()
            y = y.cuda()

        batch_size = self.batch_size
        n_samples = len(x)
        outputs = []

        for i in range(0, n_samples, batch_size):
            batch_x = x[i : i + batch_size]
            batch_y = y[i : i + batch_size]
            with torch.no_grad():
                output = self.forward(batch_x, batch_y)
            outputs.append(output)

        return torch.cat(outputs, dim=0)


class HighCapacityCMLP(MLP):
    """
    High-capacity version of CMLP with enhanced architecture and performance.
    Features:
    - Multiple hidden layers with residual connections
    - Layer normalization and advanced activation functions
    - Attention mechanism for better feature interaction
    - Efficient batch processing and memory management
    """

    def __init__(
        self,
        input_size,
        hidden_layer_size=256,
        output_size=1,
        layer_norm=True,
        drop_out=True,
        drop_out_p=0.1,
        bias=True,
        flatten=True,
        batch_size=64,
        num_layers=3,
    ):
        # Handle case where hidden_layer_size is a list
        if isinstance(hidden_layer_size, (list, tuple)):
            hidden_layer_size = hidden_layer_size[0]

        self.hidden_size = int(hidden_layer_size)

        # Initialize parent MLP first
        super(HighCapacityCMLP, self).__init__(
            input_size,
            self.hidden_size,
            self.hidden_size,
            layer_norm,
            drop_out,
            drop_out_p,
            bias,
        )

        self.num_layers = num_layers

        # Enhanced network architecture
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(torch.nn.Linear(self.hidden_size, self.hidden_size, bias=bias))

        # Advanced activation functions
        self.sigma = torch.nn.GELU()
        self.attention = torch.nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=4, batch_first=True)

        # Additional components
        self.layer_norms = torch.nn.ModuleList([torch.nn.LayerNorm(self.hidden_size) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(drop_out_p if drop_out else 0.0)

        # Final projection to output size
        self.final_layer = torch.nn.Linear(self.hidden_size, output_size, bias=bias)

        # Configuration
        self.flatten = flatten
        self.batch_size = batch_size

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for layer in self.hidden_layers:
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def _forward_features(self, x):
        """Enhanced forward pass through the feature extraction layers"""
        # Initial projection through base MLP to get [batch_size, hidden_size]
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        if self.flatten:
            x = torch.flatten(x, start_dim=1)

        x = self.model(x)  # Should now output [batch_size, hidden_size]

        # Process through multiple layers with residual connections
        for i in range(self.num_layers):
            residual = x

            # Layer normalization
            x = self.layer_norms[i](x)

            # Attention in middle layer
            if i == self.num_layers // 2:
                x_attn = x.unsqueeze(1)  # [batch_size, 1, hidden_size]
                x_attn, _ = self.attention(x_attn, x_attn, x_attn)
                x = x_attn.squeeze(1)  # [batch_size, hidden_size]

            # Dense layer with activation
            x = self.hidden_layers[i](x)
            x = self.sigma(x)
            x = self.dropout(x)

            # Residual connection
            x = x + residual

        return x

    @torch.no_grad()
    def process_batch(self, x_batch):
        """Efficient batch processing"""
        if self.flatten:
            x_batch = torch.flatten(x_batch, start_dim=1)
        return self._forward_features(x_batch)

    def forward(self, x, y) -> torch.Tensor:
        """Enhanced forward pass with optimized processing"""
        if len(x.shape) > 2 or len(y.shape) > 2:
            if self.flatten:
                x = torch.flatten(x, start_dim=1)
                y = torch.flatten(y, start_dim=1)
            else:
                x = x.reshape(-1, x.shape[-1])
                y = y.reshape(-1, y.shape[-1])

        # Get feature representations
        g_x = self._forward_features(x)  # [batch_size, hidden_size]
        g_y = self._forward_features(y)  # [batch_size, hidden_size]

        # Project to output space
        g_x = self.final_layer(g_x)  # [batch_size, output_size]
        g_y = self.final_layer(g_y)  # [batch_size, output_size]

        # Compute betting score
        score = torch.log1p(0.5 * (self.sigma(g_x) - self.sigma(g_y)))
        return score.squeeze(-1)

    def predict_large_input(self, x, y, use_gpu=False):
        """Efficient processing of large inputs with automatic memory management"""
        if use_gpu and torch.cuda.is_available():
            self.cuda()
            x = x.cuda()
            y = y.cuda()

        # Automatic batch size adjustment based on available memory
        try:
            batch_size = self.batch_size
            n_samples = len(x)
            outputs = []

            for i in range(0, n_samples, batch_size):
                batch_x = x[i : i + batch_size]
                batch_y = y[i : i + batch_size]
                with torch.no_grad():
                    output = self.forward(batch_x, batch_y)
                outputs.append(output.cpu() if use_gpu else output)

                # Clear cache periodically
                if use_gpu and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()

            return torch.cat(outputs, dim=0)

        except RuntimeError as e:  # Handle OOM errors
            if "out of memory" in str(e) and batch_size > 1:
                print(f"OOM error, reducing batch size from {batch_size} to {batch_size // 2}")
                self.batch_size = batch_size // 2
                return self.predict_large_input(x, y, use_gpu)
            else:
                raise e
