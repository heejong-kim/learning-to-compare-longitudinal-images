import copy
import torch.nn as nn
from torchvision.models import resnet18

class Resnet18Diff(nn.Module):
    def __init__(self, channels=3):
        super(Resnet18Diff, self).__init__()
        resnet = resnet18()
        if channels != 3:
            resnet.conv1 = nn.Conv2d(channels, 64, 7, 2, 3, bias=False)

        resnet.fc = nn.Identity()
        self.features = resnet
        fc = []
        fc.append(nn.Linear(512, 1, bias=False))
        self.classifier = nn.Sequential(*fc)

    def forward(self, x1, x2):
        x1 = self.features(x1)
        x2 = self.features(x2)
        x = x1 - x2
        x = self.classifier(x)
        return x

class Resnet18Regression(nn.Module):
    def __init__(self, channels=3):
        super(Resnet18Regression, self).__init__()
        resnet = resnet18()
        if channels != 3:
            resnet.conv1 = nn.Conv2d(channels, 64, 7, 2, 3, bias=False)

        resnet.fc = nn.Identity()
        self.features = resnet
        fc = []
        fc.append(nn.Linear(512, 1, bias=True))
        self.classifier = nn.Sequential(*fc)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Resnet18DiffForPairVisFeature(nn.Module):
    # TODO: rename and reflect to testing script
    def __init__(self, channels=3):
        super(Resnet18DiffForPairVisFeature, self).__init__()

        self.gradients = []
        self.activations = []

        resnet = resnet18()
        if channels != 3:
            resnet.conv1 = nn.Conv2d(channels, 64, 7, 2, 3, bias=False)

        resnet.fc = nn.Identity()

        self.features = copy.deepcopy(resnet)

        self.handles = []

        self.handles.append(
            self.features.layer4[-1].register_forward_hook(
                self.save_activation
            ))
        self.handles.append(
            self.features.layer4[-1].register_forward_hook(
                self.save_gradient
            ))


        fc = []
        fc.append(nn.Linear(512, 1, bias=False))
        self.classifier = nn.Sequential(*fc)
        self.activation_sigmoid = nn.Sigmoid()

    def forward(self, x1):
        x1 = self.features(x1)
        return x1

    def release(self):
        self.activations = []
        self.gradients = []

    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            print('no grad')
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

class Resnet18RegressionForPairVisFeature(nn.Module):
    def __init__(self, channels=3, bias=True):
        super(Resnet18RegressionForPairVisFeature, self).__init__()

        self.gradients = []
        self.activations = []

        resnet = resnet18()
        if channels != 3:
            resnet.conv1 = nn.Conv2d(channels, 64, 7, 2, 3, bias=False)

        resnet.fc = nn.Identity()

        self.features1 = copy.deepcopy(resnet)
        self.handles = []
        self.handles.append(
            self.features1.layer4[-1].register_forward_hook(
                self.save_activation
            ))
        self.handles.append(
            self.features1.layer4[-1].register_forward_hook(
                self.save_gradient
            ))

        fc = []
        fc.append(nn.Linear(512, 1, bias=bias))
        self.classifier = nn.Sequential(*fc)

    def forward(self, x):
        x = self.features1(x)
        return x

    def release(self):
        self.activations = []
        self.gradients = []

    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            print('no grad')
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)
