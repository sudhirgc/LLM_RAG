import torch

print(torch.__version__)

cuda_available = torch.cuda.is_available()
if cuda_available:
    # Print the number of GPUs detected
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Print the name of the default GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")

    # Create a tensor on the GPU and verify its device
    tensor_on_gpu = torch.randn(5, 5).cuda()
    print(f"Tensor on GPU: {tensor_on_gpu.is_cuda}")
    print(f"Tensor device: {tensor_on_gpu.device}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")