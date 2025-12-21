import torch

# 1. Check if PyTorch can see your GPU and CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # 2. Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")

    # 3. Check the CUDA version PyTorch was compiled with
    cuda_version = torch.version.cuda
    print(f"PyTorch Compiled CUDA Version: {cuda_version}")

    # 4. Test communication by moving a tensor to the GPU
    device = torch.device("cuda:0")
    test_tensor = torch.randn(2, 2).to(device)
    print(f"\nTest Tensor successfully moved to: {test_tensor.device}")
else:
    print("PyTorch cannot find the GPU. Please check your installation.")

# Exit Python interpreter
# exit()