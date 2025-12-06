"""Quick GPU detection script for YOLO POS system."""
import torch

print("=" * 50)
print("GPU Detection Check")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {gpu_props.total_memory / 1e9:.2f} GB")
    print(f"Compute capability: {gpu_props.major}.{gpu_props.minor}")
    print("\n✅ GPU is available and ready for use!")
else:
    print("\n⚠️  No GPU detected - will run on CPU")
    print("To enable GPU support, install CUDA-enabled PyTorch:")
    print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("=" * 50)
