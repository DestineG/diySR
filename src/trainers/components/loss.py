# src/trainers/components/loss.py

import torch.nn as nn

def get_loss_function(loss_config):
    loss_name = loss_config.get("lossFuncName", "l1_loss").lower()
    loss_args = loss_config.get("lossFuncArgs").get(loss_name)

    if loss_name == "l1_loss":
        loss_fn = nn.L1Loss(**loss_args)

    elif loss_name == "mse_loss":
        loss_fn = nn.MSELoss(**loss_args)

    elif loss_name == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(**loss_args)

    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

    return loss_fn