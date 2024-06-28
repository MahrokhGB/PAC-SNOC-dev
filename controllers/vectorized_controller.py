import torch, math
from config import device
from collections import OrderedDict
from pyro.distributions import Normal


# ------ vectorized module ------
# The `VectorizedModule` class in Python provides methods for managing and
# manipulating parameters of a neural network module in a vectorized manner.
class VectorizedModule:

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def parameter_shapes(self):
        return OrderedDict([(name, param.shape) for name, param in self.named_parameters().items()])

    def named_parameters(self):
        return self._params

    def _param_module(self, name, module):
        assert type(name) == str
        assert hasattr(module, 'named_parameters')

        for param_name, param in module.named_parameters().items():
            self._param(name + '.' + param_name, param)
        return module

    def _param(self, name, tensor):
        assert type(name) == str
        assert isinstance(tensor, torch.Tensor)
        assert name not in list(self._params.keys())
        if not device.type == tensor.device.type:
            tensor = tensor.to(device)
        self._params[name] = tensor
        return tensor

    def parameters(self):
        return list(self.named_parameters().values())

    def set_parameter(self, name, value):
        layer_name = name.split('.')[0]
        current_val = getattr(getattr(self, layer_name), name.split('.')[1])
        value = value.reshape(current_val.shape)
        if value.is_leaf:
            value.requires_grad=current_val.requires_grad
        setattr(getattr(self, layer_name), name.split('.')[1], value)

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():
            self.set_parameter(name, value)

    def parameters_as_vector(self):
        return torch.cat(self.parameters(), dim=-1)

    def set_parameters_as_vector(self, value):
        # value is reshaped to the parameter shape
        idx = 0
        for name, shape in self.parameter_shapes().items():
            idx_next = idx + shape[-1]
            if value.ndim == 1:
                self.set_parameter(name, value[idx:idx_next])
            elif value.ndim == 2:
                self.set_parameter(name, value[:, idx:idx_next])
            else:
                raise AssertionError
            idx = idx_next
        assert idx_next == value.shape[-1]

    def print_params(self):
        print(self.named_parameters())

    def forward(self, x):
        raise NotImplementedError


class LayerVectorized(VectorizedModule):
    def __init__(self, input_dim, output_dim, nonlinearity, requires_bias=True):
        super().__init__(input_dim, output_dim)
        self.nonlinearity = nonlinearity

        # initialize weights using the Kaiming method
        self.weight, _ = _kaiming_uniform_batched(
            torch.normal(0, 1, size=(input_dim * output_dim,),
                device=device, requires_grad=True, dtype=torch.float32
            ),
            fan=self.input_dim, a=math.sqrt(5), nonlinearity=self.nonlinearity
        )

        # initialize bias using the Kaiming method
        if requires_bias:
            fan_in = self.output_dim
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
            # bias should be initialized from a uniform dist, but since log prob might be -inf, approx by normal
            bias_dist = Normal(
                torch.zeros(output_dim,),
                torch.tensor([math.sqrt(bound*(bound+1)/3)]*output_dim
            ).float().to(device))
            self.bias = bias_dist.sample()
        else:
            self.bias = torch.zeros(
                size=(output_dim,), device=device,
                requires_grad=False, dtype=torch.float32
            )

    def forward(self, x):
        """
        The forward function performs a batched linear transformation on the input data using the
        weights and biases of a neural network layer.

        :param x: The `x` parameter in the `forward` method represents the input data that will be
        passed through the neural network layer. It can be a tensor with dimensions `[batch_size,
        input_dim]` for a single input sample or `[model_batch_size, batch_size, input_dim]` for
        multiple
        :return: The `forward` method returns the result of the linear transformation operation applied
        to the input `x`. The specific operation performed depends on the shape of the weight tensor
        `self.weight`.
        """
        if self.weight.ndim == 2 or self.weight.ndim == 3:
            model_batch_size = self.weight.shape[0]

            # batched computation
            if self.weight.ndim == 3:
                assert self.weight.shape[-2] == 1 and self.bias.shape[-2] == 1
            if len(self.bias.shape) == 1:  # if bias is 1D
                self.bias = self.bias.repeat(self.weight.shape[0], 1)

            W = self.weight.view(model_batch_size, self.output_dim, self.input_dim)
            b = self.bias.view(model_batch_size, self.output_dim)

            if x.ndim == 2:
                # introduce new dimension 0
                x = torch.reshape(x, (1, x.shape[0], x.shape[1]))
                # tile dimension 0 to model_batch size
                x = x.repeat(model_batch_size, 1, 1)
            else:
                assert x.ndim == 3 and x.shape[0] == model_batch_size
            # out dimensions correspond to [nn_batch_size, data_batch_size, out_features)
            output = torch.bmm(x.float(), W.float().permute(0, 2, 1)) + b[:, None, :].float()
        elif self.weight.ndim == 1:
            output = torch.nn.functional.linear(x, self.weight.view(self.output_dim, self.input_dim), self.bias)
        else:
            raise NotImplementedError
        # apply nonlinearity
        if self.nonlinearity is not None:
            output = self.nonlinearity(output)
        return output

    def parameter_shapes(self):
        if self.bias.requires_grad:
            return OrderedDict(bias=self.bias.shape, weight=self.weight.shape)
        else:
            return OrderedDict(weight=self.weight.shape)

    def named_parameters(self):
        if self.bias.requires_grad:
            return OrderedDict(bias=self.bias, weight=self.weight)
        else:
            return OrderedDict(weight=self.weight)

    def __call__(self, *args, **kwargs):
        return self.forward( *args, **kwargs)


class VectorizedController(VectorizedModule):
    """Trainable neural network that batches multiple sets of parameters.
    """
    def __init__(self, num_states, num_inputs, layer_sizes=(64, 64),  nonlinearity_hidden=torch.tanh,
                 nonlinearity_output=torch.tanh, requires_bias={'out':True, 'hidden':True}):
        # requires_bias: add bias to the hidden and output layers or not
        super().__init__(input_dim=num_states, output_dim=num_inputs)

        self.nonlinearity_hidden = nonlinearity_hidden
        self.nonlinearity_output = nonlinearity_output
        self.n_layers = len(layer_sizes)
        prev_size = num_states
        for i, size in enumerate(layer_sizes):
            setattr(
                self, 'fc_%i'%(i+1),
                LayerVectorized(
                    prev_size, size,
                    requires_bias=requires_bias['hidden'],
                    nonlinearity=self.nonlinearity_hidden))
            prev_size = size
        setattr(
            self, 'out',
            LayerVectorized(
                prev_size, num_inputs,
                requires_bias=requires_bias['out'],
                nonlinearity=self.nonlinearity_output))

    def forward(self, x):
        """
        The `forward` function takes an input `x`, passes it through a series of fully connected layers
        and non-linear activation functions, and returns the final output.
        """
        output = x
        # apply hidden layers
        for i in range(1, self.n_layers + 1):
            output = getattr(self, 'fc_%i' % i)(output)
        # apply output layer
        output = getattr(self, 'out')(output)
        return output

    def parameter_shapes(self):
        param_dict = OrderedDict()
        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).parameter_shapes().items():
                param_dict[layer_name + '.' + name] = param
        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).parameter_shapes().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def named_parameters(self):
        param_dict = OrderedDict()
        # hidden layers
        for i in range(1, self.n_layers + 1):
            layer_name = 'fc_%i' % i
            for name, param in getattr(self, layer_name).named_parameters().items():
                param_dict[layer_name + '.' + name] = param
        # last layer
        layer_name = 'out'
        for name, param in getattr(self, layer_name).named_parameters().items():
            param_dict[layer_name + '.' + name] = param

        return param_dict

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ------ linear vectorized ------
class LinearController(VectorizedController):
    def __init__(self, num_states, num_inputs, requires_bias={'out':True, 'hidden':True}):
        super().__init__(
            num_states, num_inputs, layer_sizes=[], nonlinearity_hidden=None,
            nonlinearity_output=None, requires_bias=requires_bias
        )

# Initialization Helpers
def _kaiming_uniform_batched(tensor, fan, a=0.0, nonlinearity=torch.tanh):
    nonlinearity = 'linear' if nonlinearity == None else nonlinearity.__name__
    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound), Normal(torch.tensor([0]*tensor.size(dim=0)).float().to(device),
                                                      torch.tensor([math.sqrt(bound*(bound+1)/3)]*tensor.size(dim=0)).float().to(device))

