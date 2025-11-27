# src/trainers/components/__init__.py

from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .loss import get_loss_function

def get_train_components(model_parameters,
                         optimizer_config, optimizer_state,
                         scheduler_config, scheduler_state,
                         loss_config):
    optimizer = get_optimizer(model_parameters, optimizer_state, optimizer_config)
    scheduler = get_scheduler(optimizer, scheduler_state, scheduler_config)
    loss_fn = get_loss_function(loss_config)

    return optimizer, scheduler, loss_fn