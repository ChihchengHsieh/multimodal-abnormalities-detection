import torch
import torch.nn as nn

from models.setup import ModelSetup

def get_optimiser(model: nn.Module , setup: ModelSetup):

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Model size: {sum([param.nelement()  for param in model.parameters()]):,}")

    if setup.optimiser == 'adamw':
        print(f"Using AdamW as optimizer with lr={setup.lr}")
        optimiser = torch.optim.AdamW(
            params, lr=setup.lr, betas=(0.9, 0.999), weight_decay=setup.weight_decay,
        )

    elif setup.optimiser == 'sgd':
        print(f"Using SGD as optimizer with lr={setup.lr}")
        optimiser = torch.optim.SGD(
            params,
            lr=setup.lr,
            momentum=0.9,
            weight_decay=setup.weight_decay,
        )
    else:
        raise Exception(f"Unsupported optimiser {setup.optimiser}")

    return optimiser
