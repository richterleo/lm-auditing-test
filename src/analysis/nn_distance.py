import importlib
import os
import torch
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
src_path = os.path.join(project_root, "src")
if project_root not in sys.path:
    sys.path.append(project_root)
if src_path not in sys.path:
    sys.path.append(src_path)

orig_models = importlib.import_module("deep-anytime-testing.models.mlp", package="deep-anytime-testing")
MLP = getattr(orig_models, "MLP")


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
        """
        Initializes the MMDEMLP object.

        Args:
        - input_size (int): Size of input layer.
        - hidden_layer_size (int): Size of hidden layer.
        - output_size (int): Size of output layer.
        - layer_norm (bool): Indicates if layer normalization should be applied.
        - drop_out (bool): Indicates if dropout should be applied.
        - drop_out_p (float): The dropout probability, i.e., the probability of an element to be zeroed.
        - bias (bool): If set to False, the layers will not learn an additive bias.
        - flatten (bool): Determines if the input tensors should be flattened.
        """

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

        # Activation function for the custom operation in the forward method
        self.sigma = torch.nn.Tanh()
        self.flatten = flatten

    def forward(self, x, y) -> torch.Tensor:
        """
        Forward pass for the MMDEMLP model. Computes the output based on inputs x and y.

        Args:
        - x (torch.Tensor): First input tensor.
        - y (torch.Tensor): Second input tensor.

        Returns:
        - torch.Tensor: The output of the forward pass.
        """

        # If input tensors have more than two dimensions
        if len(x.shape) > 2 or len(y.shape) > 2:
            if self.flatten:
                # Flatten the tensors from dimension 1
                x = torch.flatten(x, start_dim=1)
                y = torch.flatten(y, start_dim=1)
                g_x = self.model(x)
                g_y = self.model(y)
            else:
                num_samples = x.shape[-1]
                g_x, g_y = 0, 0
                # Process each sample in the tensor
                for i in range(num_samples):
                    g_x += self.model(torch.flatten(x[..., i], start_dim=1)) / num_samples
                    g_y += self.model(torch.flatten(y[..., i], start_dim=1)) / num_samples
        else:
            # If tensors are two-dimensional
            g_x = self.model(x)
            g_y = self.model(y)

        # Compute the custom output based on the difference of outputs
        # ! This is a more direct implementation of the custom operation
        output = torch.log(1 + 0.5 * self.sigma(g_x) - 0.5 * self.sigma(g_y))

        return output
