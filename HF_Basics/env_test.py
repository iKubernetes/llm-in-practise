import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))

print(torch.version.cuda)
print(torch.version.git_version)
print(torch.__version__)

