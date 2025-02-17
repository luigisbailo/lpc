import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    A class representing a classifier neural network.
    """

    def __init__(self, architecture_type, architecture, num_classes, penultimate_nodes):
        """
        Args:
            architecture_type (str): Type of model ('lin_pen', 'no_pen', 'nonlin_pen').
            architecture (dict): Architecture configuration.
            num_classes (int): Number of output classes.
            penultimate_nodes (int): Number of nodes in the penultimate layer.
        """
        super().__init__()

        self.architecture_type = architecture_type
        self.architecture = architecture
        self.penultimate_nodes = penultimate_nodes
        self.architecture_hypers = architecture['hypers']
        self.nodes_head = architecture['hypers'].get('nodes_head', [])
        self.num_classes = num_classes

        # Retrieve the activation class dynamically from nn.
        activation_class = getattr(nn, self.architecture_hypers['activation'])
        self.activation = activation_class()

        if architecture_type == 'lin_pen':
            self.penultimate_linear_nodes = penultimate_nodes
            self.penultimate_nonlinear_nodes = None
        elif architecture_type == 'no_pen':
            self.penultimate_linear_nodes = None
            self.penultimate_nonlinear_nodes = None
        elif architecture_type == 'nonlin_pen':
            self.penultimate_linear_nodes = None
            self.penultimate_nonlinear_nodes = penultimate_nodes
        else:
            raise ValueError(f"Unsupported architecture name: {architecture_type}")

    def reset_parameters(self):
        """Resets the parameters of all layers."""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def make_head(self, input_dims_head):
        """
        Creates the head layers of the classifier.

        Args:
            input_dims_head (int): Input dimensions for the head layers.
        """
        self.dropout_head = self.architecture_hypers.get('dropout_head', None)
        self.bn_head = self.architecture_hypers.get('bn_head', False)

        if self.dropout_head and len(self.dropout_head) != len(self.nodes_head):
            raise ValueError('The length of dropout values must match the number of head layers.')

        layers = []
        layer_dims = [input_dims_head] + self.nodes_head

        for i in range(len(layer_dims) - 1):
            if self.dropout_head:
                layers.append(nn.Dropout(p=self.dropout_head[i]))
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(self.activation)
            if self.bn_head:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))

        self.head = nn.Sequential(*layers)

    def make_penultimate(self, input_dims):
        """
        Creates the penultimate layer and the output layer.

        Args:
            input_dims (int): Input dimensions for the penultimate layer.

        Returns:
            tuple: (penultimate_layer, output_layer)
        """
        if self.penultimate_linear_nodes:
            penultimate_layer = nn.Linear(input_dims, self.penultimate_linear_nodes)
            penultimate_dim = self.penultimate_linear_nodes
        elif self.penultimate_nonlinear_nodes:
            penultimate_layer = nn.Sequential(
                nn.Linear(input_dims, self.penultimate_nonlinear_nodes),
                self.activation
            )
            penultimate_dim = self.penultimate_nonlinear_nodes
        else:
            penultimate_layer = None
            penultimate_dim = input_dims

        if self.architecture_hypers.get('dropout_penultimate', False):
            dropout = nn.Dropout(p=0.5)
            output_layer = nn.Sequential(
                dropout,
                nn.Linear(penultimate_dim, self.num_classes, bias=False)
            )
        else:
            output_layer = nn.Linear(penultimate_dim, self.num_classes, bias=False)

        return penultimate_layer, output_layer

    def classifier_forward(self, x):
        """
        Performs the forward pass through the classifier components.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: A dictionary with keys 'x_output', 'x_penultimate', and 'x_backbone'.
        """
        output_dict = {}

        if self.nodes_head:
            x = self.head(x)
        x_backbone = x
        output_dict['x_backbone'] = x_backbone

        if self.penultimate_linear_nodes or self.penultimate_nonlinear_nodes:
            x_pen = self.penultimate_layer(x_backbone)
        else:
            x_pen = x_backbone
        output_dict['x_penultimate'] = x_pen

        # Ensure output is shaped as (batch_size, num_classes)
        output_dict['x_output'] = self.output_layer(x_pen).reshape(-1, self.num_classes)
        return output_dict


class MLPvanilla(Classifier):
    """
    Multi-Layer Perceptron (MLP) vanilla implementation.
    """

    def __init__(self, architecture_type, architecture, num_classes, input_dims, penultimate_nodes):
        """
        Args:
            architecture_type (str): Architecture type.
            architecture (dict): Architecture configuration.
            num_classes (int): Number of output classes.
            input_dims (int): Input dimensions.
            penultimate_nodes (int): Nodes in the penultimate layer.
        """
        super().__init__(architecture_type, architecture, num_classes, penultimate_nodes)
        self.make_head(input_dims_head=input_dims)
        head_output_dim = self.nodes_head[-1] if self.nodes_head else input_dims
        self.penultimate_layer, self.output_layer = self.make_penultimate(head_output_dim)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Output dictionary from classifier_forward.
        """
        x = torch.flatten(x, start_dim=1)
        return self.classifier_forward(x)


class BasicBlock(nn.Module):
    """
    A basic building block for a residual network.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, activation, base_width=1, stride=1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (nn.Module): Activation function.
            base_width (int): Base width multiplier.
            stride (int): Stride for the convolution.
        """
        super().__init__()
        width = base_width * out_channels
        self.activation = activation

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(width),
            self.activation,
            nn.Conv2d(width, out_channels * self.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        x_out = self.residual_function(x) + self.shortcut(x)
        return self.activation(x_out)


class Bottleneck(nn.Module):
    """
    Bottleneck block for a ResNet network.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, activation, base_width=1, stride=1):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (nn.Module): Activation function.
            base_width (int): Base width multiplier.
            stride (int): Stride for the convolution.
        """
        super().__init__()
        width = base_width * out_channels
        self.activation = activation

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width),
            self.activation,
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(width),
            self.activation,
            nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        x_out = self.residual_function(x) + self.shortcut(x)
        return self.activation(x_out)


# Mappings for ResNet models.
RESNET_BLOCK = {
    '18': BasicBlock,
    '34': BasicBlock,
    '50': Bottleneck,
    '101': Bottleneck,
    '151': Bottleneck,
}

RESNET_LAYERS = {
    '18': [2, 2, 2, 2],
    '34': [3, 4, 6, 3],
    '50': [3, 4, 6, 3],
    '101': [3, 4, 23, 3],
    '151': [3, 8, 36, 3]
}


class ResNet(Classifier):
    """
    ResNet class for image classification using Residual Networks.
    This version chooses the initial convolution type based on the input spatial resolution.
    """

    def __init__(self, architecture_type, architecture, num_classes, penultimate_nodes, input_dims):
        """
        Args:
            architecture_type (str): Architecture type.
            architecture (dict): Architecture configuration (should include 'backbone_model').
            num_classes (int): Number of output classes.
            penultimate_nodes (int): Nodes in the penultimate layer.
            input_dims (int): Spatial resolution of the input image (assumed square). 
        """
        super().__init__(architecture_type, architecture, num_classes, penultimate_nodes)
        self.in_channels = 64
        self.input_dims = input_dims 
        
        backbone_model = str(self.architecture['backbone_model'])
        if backbone_model not in RESNET_BLOCK:
            raise ValueError(f"Backbone model {backbone_model} not recognized.")
        block = RESNET_BLOCK[backbone_model]
        layers_config = RESNET_LAYERS[backbone_model]
        expansion = block.expansion

        self.make_backbone_layers(block, layers_config)

        # The head is built based on the output of the backbone.
        # Regardless of input dims, the final feature vector is 512 * expansion.
        if self.nodes_head:
            self.make_head(input_dims_head=512 * expansion)
            head_output_dim = self.nodes_head[-1]
            self.penultimate_layer, self.output_layer = self.make_penultimate(head_output_dim)
        else:
            self.penultimate_layer, self.output_layer = self.make_penultimate(512 * expansion)

    def make_layer(self, block, out_channels, num_blocks, stride=1):
        """
        Creates a sequential layer composed of residual blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, self.activation, base_width=1, stride=s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def make_backbone_layers(self, block, blocks):
        """
        Constructs the backbone layers for the ResNet.

        The initial convolutional layer is chosen based on the input image resolution:
        """
        # Use the input_dims (spatial resolution) to decide the convolution type.
        print(self.input_dims)
        if self.input_dims >= 64:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.in_channels),
                self.activation,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.in_channels),
                self.activation,
            )
        self.layer1 = self.make_layer(block, 64, blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Forward pass through the ResNet network.
        """
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier_forward(x)