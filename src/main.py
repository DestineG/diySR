# src/main.py

from src.data import get_dataloader
from src.models import get_model_by_name

if __name__ == "__main__":
    dataloader = get_dataloader()
    model = get_model_by_name('base_model')()
    # x = {hr, lr, scale}
    for batch in dataloader:
        out = model(batch)
        print(batch.keys())
        print("LR shape:", batch['lr'].shape)
        print("HR shape:", batch['hr'].shape)
        print("Encoded shape:", out['encoded'].shape)
        print("Decoded shape:", out['decoded'].shape)
        print("Out base shape:", out['out_base'].shape)
        break