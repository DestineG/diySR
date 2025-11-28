# src/trainers/components/scheduler.py

import torch.optim.lr_scheduler as lr_scheduler

def get_scheduler(optimizer, scheduler_state, config):
    scheduler_name = config.get("schedulerFuncName", "step_lr").lower()
    scheduler_args = config.get("schedulerFuncArgs")

    if scheduler_name == "step_lr":
        scheduler = lr_scheduler.StepLR(optimizer, **scheduler_args)

    elif scheduler_name == "exponential_lr":
        scheduler = lr_scheduler.ExponentialLR(optimizer, **scheduler_args)

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    return scheduler