import torch
import torch.nn as nn
import torch.nn.functional as F

class LRPConv1D(nn.Module):
    def __init__(self, layer, epsilon=1e-6):
        super(LRPConv1D, self).__init__()
        self.layer = layer
        self.epsilon = epsilon
    
    def forward(self, input, relevance):
        with torch.no_grad():
            z = self.layer(input) + self.epsilon * torch.sign(self.layer(input))  # Stabilization
            s = relevance / z  # Element-wise division
            c = torch.autograd.grad(z, input, s)[0]  # Gradient-based propagation
            return input * c


class LRPLinear(nn.Module):
    def __init__(self, layer, epsilon=1e-6):
        super(LRPLinear, self).__init__()
        self.layer = layer
        self.epsilon = epsilon
    
    def forward(self, input, relevance):
        with torch.no_grad():
            z = self.layer(input) + self.epsilon * torch.sign(self.layer(input))
            s = relevance / z
            c = torch.autograd.grad(z, input, s)[0]
            return input * c


class LRPPooling(nn.Module):
    def __init__(self, layer):
        super(LRPPooling, self).__init__()
        self.layer = layer
    
    def forward(self, input, relevance):
        with torch.no_grad():
            z = self.layer(input)
            s = relevance / (z + 1e-6)  # Stabilization for division
            c = torch.autograd.grad(z, input, s)[0]
            return input * c


# def apply_lrp(model, input_tensor, epsilon=1e-6):
#     relevance = model(input_tensor, return_activations=True)[0]  # Get output relevance
    
#     for name, layer in reversed(list(model.named_children())):
#         if isinstance(layer, nn.Conv1d):
#             relevance = LRPConv1D(layer, epsilon)(input_tensor, relevance)
#         elif isinstance(layer, nn.Linear):
#             relevance = LRPLinear(layer, epsilon)(input_tensor, relevance)
#         elif isinstance(layer, nn.AdaptiveAvgPool1d):
#             relevance = LRPPooling(layer)(input_tensor, relevance)
    
#     return relevance

def apply_lrp(model, input_tensor, epsilon=1e-6):
    # Forward pass with activations
    outputs, *activations = model(input_tensor, return_activations=True)

    relevance = outputs  # Start LRP from the model output

    for name, layer in reversed(list(model.named_children())):
        print(f"Processing layer: {name}, Type: {type(layer)}, Relevance shape: {relevance.shape}")  # Debugging

        activation = activations.pop()  # Get corresponding activation

        if isinstance(layer, nn.Linear):
            if activation.shape != relevance.shape:
                print(f"⚠️ Shape mismatch! Reshaping relevance {relevance.shape} -> {activation.shape}")
                relevance = relevance.view_as(activation)  # Match the activation shape

            relevance = LRPLinear(layer, epsilon)(activation, relevance)

        elif isinstance(layer, nn.Conv1d):
            relevance = LRPConv1D(layer, epsilon)(activation, relevance)

        elif isinstance(layer, nn.AdaptiveAvgPool1d):
            relevance = LRPPooling(layer)(activation, relevance)

    return relevance


# def apply_lrp(model, input_tensor, epsilon=1e-6):
#     # Forward pass with activations
#     outputs = model(input_tensor, return_activations=True)
#     relevance = outputs[0]  # Output relevance

#     # Fully connected layers
#     relevance = LRPLinear(model.fc3, epsilon)(outputs[-1], relevance)
#     relevance = LRPLinear(model.fc2, epsilon)(outputs[-2], relevance)
#     relevance = LRPLinear(model.fc1, epsilon)(outputs[-3], relevance)

#     # Global pooling layer
#     relevance = LRPPooling(model.global_pool)(outputs[-4], relevance)

#     # Split relevance across convolution branches
#     branch_relevance = torch.chunk(relevance, 4, dim=1)  # Split into 4 parts

#     # Apply LRP to each convolutional branch separately
#     branch1x1_relevance = LRPConv1D(model.branch1x1[0], epsilon)(outputs[1], branch_relevance[0])
#     branch3x3_relevance = LRPConv1D(model.branch3x3[1], epsilon)(outputs[2], branch_relevance[1])
#     branch5x5_relevance = LRPConv1D(model.branch5x5[1], epsilon)(outputs[3], branch_relevance[2])
#     branch_pool_relevance = LRPConv1D(model.branch_pool[1], epsilon)(outputs[4], branch_relevance[3])

#     # Sum up all branch relevance to match the input shape
#     relevance = branch1x1_relevance + branch3x3_relevance + branch5x5_relevance + branch_pool_relevance

#     return relevance