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

def test_cnn():
    try:
        import torch
        x = torch.randn(1, 3, 224, 224).cuda()
        conv = torch.nn.Conv2d(3, 3, 3).cuda()
        out = conv(x)
        print(out.sum())
    except Exception as e:
        print(e)


def test_cnn_pretrained_mode_inference():
    # deeplabv3_resnet50

    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(pretrained=False, num_classes=7).to('cuda')

    # try to run the model
    x = torch.randn(3, 3, 256, 256).cuda()
    try:
        out = model(x)
        print(out)  
    except Exception as e:
        print(e)

def test_cnn_pretrained_mode_training():
    # deeplabv3_resnet50

    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(pretrained=False, num_classes=7).to('cuda')
    
    # try to train the model
    x = torch.randn(3, 3, 256, 256).cuda()
    y = torch.randint(0, 7, (3, 256, 256)).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    try:
        for i in range(10):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out['out'], y)
            loss.backward()
            optimizer.step()
            print(loss.item())
    except Exception as e:
        print(e)

def test_cnn_pretrained_mode_training_2():
    # resnet 50

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import models

    model = models.resnet50(pretrained=False).to('cuda')
    model.fc = nn.Linear(2048, 7).to('cuda')

    # try to train the model
    x = torch.randn(3, 3, 256, 256).cuda()
    y = torch.randint(0, 7, (3,)).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        for i in range(10):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            print(loss.item())
    except Exception as e:
        print(e)

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
    test_cnn()
    test_cnn_pretrained_mode_inference()
    test_cnn_pretrained_mode_training()
    test_cnn_pretrained_mode_training_2()
    cuda_matrix_multiplication_benchmark()
    cpu_matrix_multiplication_benchmark()
