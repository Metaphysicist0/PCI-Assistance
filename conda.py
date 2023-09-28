import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))