import logging

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from numpy.f2py.auxfuncs import throw_error

from siclib.models.base_model import BaseModel
from siclib.models.utils.modules import ConvModule

logger = logging.getLogger(__name__)

# flake8: noqa
# mypy: ignore-errors


class EnhanceLowLevelEncoder(BaseModel):
    default_conf = {
        "feat_dim": 64,
        "in_channel": 3,
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        logger.debug(f"Initializing LowLevelEncoder with {conf}")

        self.conv1 = ConvModule(conf.in_channel, conf.feat_dim, kernel_size=3, padding=1)
        self.conv2 = ConvModule(conf.feat_dim, conf.feat_dim, kernel_size=5, padding=2)
        self.conv3 = ConvModule(conf.feat_dim, conf.feat_dim, kernel_size=7, padding=3)

    def _apply_laplacian(self, x):
        """实现拉普拉斯卷积"""
        pad_x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        sharpened = (
                -1 * pad_x[..., :-2, :-2]
                - 1 * pad_x[..., :-2, 1:-1]
                - 1 * pad_x[..., :-2, 2:]
                + -1 * pad_x[..., 1:-1, :-2]
                + 8 * x
                - 1 * pad_x[..., 1:-1, 2:]
                + -1 * pad_x[..., 2:, :-2]
                - 1 * pad_x[..., 2:, 1:-1]
                - 1 * pad_x[..., 2:, 2:]
        )
        return sharpened

    def _forward(self, data):
        x = data["image"]

        assert (
            x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0
        ), "Image size must be multiple of 32 if not using single image input."

        out = self._apply_laplacian(x)
        for i in range(3):
            plt.imshow(out[0][i].detach().cpu(),cmap="gray")
            plt.show()
        out = self.conv1(out)
        out = self.conv2(out)
        for i in range(32):
            plt.imshow(out[0][i].detach().cpu(),cmap="gray")
            plt.show()
        out = self._apply_laplacian(out)
        for i in range(32):
            plt.imshow(out[0][i].detach().cpu(),cmap="gray")
            plt.show()
        out = self.conv3(out)
        for i in range(32):
            plt.imshow(out[0][i].detach().cpu(),cmap="gray")
            plt.show()

        return {"features": out}

    def loss(self, pred, data):
        raise NotImplementedError