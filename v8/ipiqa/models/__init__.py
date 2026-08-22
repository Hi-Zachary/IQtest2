"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import torch
from omegaconf import OmegaConf
from ipiqa.common.registry import registry

from ipiqa.models.base_model import BaseModel

from ipiqa.models.model_v8 import MSQRNetV8
from ipiqa.models.model_v9 import MSQRNetV9
from ipiqa.models.model_v10 import MSQRNetV10
from ipiqa.models.model_v11 import MSQRNetV11
from ipiqa.models.model_v12 import MSQRNetV12
from ipiqa.models.model_v14 import MSQRNetV14
from ipiqa.models.model_v15 import MSQRNetV15
from ipiqa.models.model_v16 import MSQRNetV16
from ipiqa.models.model_v17 import MSQRNetV17
from ipiqa.models.model_v18 import MSQRNetV18
from ipiqa.models.model_v19 import MSQRNetV19
from ipiqa.models.model_v20 import MSQRNetV20
from ipiqa.models.model_v21 import MSQRNetV21
from ipiqa.models.model_v23 import MSQRNetV23


__all__ = [
    "load_model",
    "BaseModel",
    "MSQRNetV8",
    "MSQRNetV9",
    "MSQRNetV10",
    "MSQRNetV11",
    "MSQRNetV12",
    "MSQRNetV14",
    "MSQRNetV15",
    "MSQRNetV16",
    "MSQRNetV17",
    "MSQRNetV18",
    "MSQRNetV19",
    "MSQRNetV20",
    "MSQRNetV21",
    "MSQRNetV23",
]


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    """
    Load supported models.

    To list all available models and types in registry:
    >>> from lavis.models import model_zoo
    >>> print(model_zoo)

    Args:
        name (str): name of the model.
        model_type (str): type of the model.
        is_eval (bool): whether the model is in eval mode. Default: False.
        device (str): device to use. Default: "cpu".
        checkpoint (str): path or to checkpoint. Default: None.
            Note that expecting the checkpoint to have the same keys in state_dict as the model.

    Returns:
        model (torch.nn.Module): model.
    """

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)
