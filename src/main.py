# src/main.py

from src.data import get_dataloader

if __name__ == "__main__":
    dataloader = get_dataloader()
    for batch in dataloader:
        print(batch)
        break  # 仅打印第一个批次以验证加载器工作正常
