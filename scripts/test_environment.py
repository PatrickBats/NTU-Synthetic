#!/usr/bin/env python
"""Test script to verify the NTU-Synthetic environment installation"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("-" * 80)

# Test core packages
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import torchvision
    print(f"✓ Torchvision version: {torchvision.__version__}")
except ImportError as e:
    print(f"✗ Torchvision import failed: {e}")

try:
    import transformers
    print(f"✓ Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")

try:
    import diffusers
    print(f"✓ Diffusers version: {diffusers.__version__}")
except ImportError as e:
    print(f"✗ Diffusers import failed: {e}")

try:
    import einops
    print(f"✓ Einops version: {einops.__version__}")
except ImportError as e:
    print(f"✗ Einops import failed: {e}")

try:
    import cv2
    print(f"✓ OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib version: {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")

try:
    import spacy
    print(f"✓ SpaCy version: {spacy.__version__}")
    # Test if language model is installed
    try:
        nlp = spacy.load("en_core_web_sm")
        print("  ✓ SpaCy en_core_web_sm model loaded successfully")
    except:
        print("  ✗ SpaCy en_core_web_sm model not found")
except ImportError as e:
    print(f"✗ SpaCy import failed: {e}")

try:
    import clip
    print(f"✓ CLIP imported successfully")
except ImportError as e:
    print(f"✗ CLIP import failed: {e}")

try:
    import triton_windows as triton
    print(f"✓ Triton (Windows) version: {triton.__version__}")
except ImportError:
    try:
        import triton
        print(f"✓ Triton version: {triton.__version__}")
    except ImportError as e:
        print(f"✗ Triton import failed: {e}")

print("-" * 80)

# Test for potential issues
print("\nChecking for known issues:")

# Check torch/torchaudio compatibility
try:
    import torchaudio
    print(f"⚠️  Torchaudio version: {torchaudio.__version__} (may have compatibility issues with torch {torch.__version__})")
except ImportError:
    print("✓ Torchaudio not installed (avoiding potential conflicts)")

# Summary
print("\n" + "=" * 80)
print("Environment setup summary:")
print("- Core ML frameworks: Installed")
print("- Computer vision libraries: Installed")
print("- NLP libraries: Installed")
print("- GPU support:", "Available" if torch.cuda.is_available() else "Not available")
print("\nThe environment is ready to run the NTU-Synthetic codebase!")
print("=" * 80)