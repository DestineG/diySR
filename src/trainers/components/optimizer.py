# src/trainers/components/optimizer.py

import torch.optim as optim

def get_optimizer(model_parameters, optimizer_state, config):
    lr = config.get("lr", 1e-3)
    optimizer_name = config.get("optimizerFuncName", "adam").lower()
    optimizer_args = config.get("optimizerFuncArgs").get(optimizer_name)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model_parameters, **optimizer_args)

    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model_parameters, **optimizer_args)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    return optimizer
