import importlib
import os
import torch
import sys

# Add the submodule and models to the path for eval_trainer
submodule_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "deep-anytime-testing"))
models_path = os.path.join(submodule_path, "models")

for path in [submodule_path, models_path]:
    if path not in sys.path:
        sys.path.append(path)

orig_models = importlib.import_module(
        "deep-anytime-testing.models.mlp", package="deep-anytime-testing"
    )
MMDEMLP = getattr(orig_models, "MMDEMLP")


class TMLP(MMDEMLP):
    """
    TMLP (tolerance) is a subclass of MMDEMLP that introduces the epsilon parameter
    and modifies the output calculation. For epsilon=0, the output is equivalent to MMDEMLP.
    """

    def __init__(self, input_size, hidden_layer_size, output_size, layer_norm, drop_out, drop_out_p, bias,
                 flatten=True, epsilon=1e-5):
        """
        Initializes the TMLP object.

        Args:
        - input_size (int): Size of input layer.
        - hidden_layer_size (int): Size of hidden layer.
        - output_size (int): Size of output layer.
        - layer_norm (bool): Indicates if layer normalization should be applied.
        - drop_out (bool): Indicates if dropout should be applied.
        - drop_out_p (float): The dropout probability, i.e., the probability of an element to be zeroed.
        - bias (bool): If set to False, the layers will not learn an additive bias.
        - flatten (bool): Determines if the input tensors should be flattened.
        - epsilon (float): The epsilon hyperparameter for the custom output calculation.
        """
        super(TMLP, self).__init__(input_size, hidden_layer_size, output_size, layer_norm, drop_out,
                                              drop_out_p, bias, flatten)
        self.epsilon = epsilon

    def forward(self, x, y) -> torch.Tensor:
        """
        Forward pass for the TMLP model. Computes the output based on inputs x and y,
        incorporating the epsilon parameter.

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

        # Compute the custom output based on the difference of outputs and incorporating epsilon
        output = torch.log(1 + self.sigma(g_x - g_y) / torch.exp(torch.tensor(self.epsilon)))

        return output

