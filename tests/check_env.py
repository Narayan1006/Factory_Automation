import torch
import sklearn
import pytest

print(f"Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"Sklearn: {sklearn.__version__}")
print(f"Pytest: {pytest.__version__}")
