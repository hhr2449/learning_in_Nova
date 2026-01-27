import torch

def main():
    print(torch.cuda.nccl.version())
    print(torch.cuda.device_count())
    
if __name__ == "__main__":
    main()