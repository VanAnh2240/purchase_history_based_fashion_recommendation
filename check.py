# pip uninstall torch torchvision torchaudio -y
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

import torch
import sys
import platform
import subprocess


def main():
    # Python + OS
    print("Python:", sys.version.split()[0])
    print("OS:", platform.system(), platform.release())

    # Torch + GPU
    print("\n--- PyTorch ---")
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        x = torch.rand(2, 2).cuda()
        print("GPU test tensor OK")


if __name__ == "__main__":
    main()