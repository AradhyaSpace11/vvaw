import torch


def main():
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"active device: {idx} - {torch.cuda.get_device_name(idx)}")
        x = torch.randn((2048, 2048), device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"test tensor device: {y.device}")


if __name__ == "__main__":
    main()
