import logging

import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.pyplot import axhspan

from siclib.models.base_model import BaseModel
from siclib.models.utils.modules import ConvModule

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class LowLevelEncoder(BaseModel):
    default_conf = {
        "feat_dim": 64,
        "in_channel": 3,
        "keep_resolution": True,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        logger.debug(f"Initializing LowLevelEncoder with {conf}")

        if self.conf.keep_resolution:
            self.conv1 = ConvModule(conf.in_channel, conf.feat_dim, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = ConvModule(conf.feat_dim, conf.feat_dim, kernel_size=5, padding=2)
            self.relu2 = nn.ReLU()
            self.conv3 = ConvModule(conf.feat_dim, conf.feat_dim, kernel_size=7, padding=3)
        else:
            self.conv1 = nn.Conv2d(
                conf.in_channel, conf.feat_dim, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(conf.feat_dim)
            self.relu = nn.ReLU(inplace=True)

    def _forward(self, data):
        x = data["image"]

        assert (
            x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0
        ), "Image size must be multiple of 32 if not using single image input."

        if self.conf.keep_resolution:
            x = self.conv1(x)
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 24))
            axes_flat = axes.flatten()
            for i, ax in enumerate(axes_flat):
                ax.imshow(x[0][i].detach().cpu(), cmap="gray")
                ax.axis('off')
            plt.show()
            x = self.relu1(x)
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 24))
            axes_flat = axes.flatten()
            for i, ax in enumerate(axes_flat):
                ax.imshow(x[0][i].detach().cpu(), cmap="gray")
                ax.axis('off')
            plt.show()
            x = self.conv2(x)
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 24))
            axes_flat = axes.flatten()
            for i, ax in enumerate(axes_flat):
                ax.imshow(x[0][i].detach().cpu(), cmap="gray")
                ax.axis('off')
            plt.show()
            x = self.relu2(x)
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 24))
            axes_flat = axes.flatten()
            for i, ax in enumerate(axes_flat):
                ax.imshow(x[0][i].detach().cpu(), cmap="gray")
                ax.axis('off')
            plt.show()
            out = self.conv3(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            out = self.relu(x)

        return {"features": out}

    def loss(self, pred, data):
        raise NotImplementedError
