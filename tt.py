import torch
import time
t = torch.tensor([0, 2, 3, 4])
m = t>0
n = torch.tensor([1, 2])
print(t[n])
# n[m] = torch.tensor([0, 0, 0, 0])[m]
# print(n)