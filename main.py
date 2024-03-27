import torch

def test_installation():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
    else:
        print("CUDA is not available.")

def cuda_matrix_multiplication_benchmark():
    import time
    import torch

    # Define the matrix size
    N = 2048

    # Create random matrices
    A = torch.randn(N, N).cuda()
    B = torch.randn(N, N).cuda()

    # Benchmark the matrix multiplication
    start_time = time.time()
    C = torch.mm(A, B)
    elapsed_time = time.time() - start_time

    print("GPU Matrix multiplication time: {:.4f} seconds".format(elapsed_time))

def cpu_matrix_multiplication_benchmark():
    import time
    import torch

    # Define the matrix size
    N = 2048

    # Create random matrices
    A = torch.randn(N, N)
    B = torch.randn(N, N)

    # Benchmark the matrix multiplication
    start_time = time.time()
    C = torch.mm(A, B)
    elapsed_time = time.time() - start_time

    print("CPU Matrix multiplication time: {:.4f} seconds".format(elapsed_time))

if __name__ == "__main__":
    test_installation()
    cuda_matrix_multiplication_benchmark()
    cpu_matrix_multiplication_benchmark()
