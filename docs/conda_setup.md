# Conda Environment Setup for NTU-Synthetic

## Quick Setup

1. **Create environment**
   ```bash
   conda create -n ntu-synthetic python=3.10 -y
   ```

2. **Install PyTorch with CUDA**
   ```bash
   conda run -n ntu-synthetic pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install dependencies**
   ```bash
   conda run -n ntu-synthetic pip install -r requirements_conda.txt
   conda run -n ntu-synthetic pip uninstall torchaudio -y
   ```

4. **Install SpaCy model**
   ```bash
   conda run -n ntu-synthetic python -m spacy download en_core_web_sm
   ```

5. **Activate and use**
   ```bash
   conda activate ntu-synthetic
   ```

## Verify Installation
```bash
python test_environment.py
```

## Notes
- Uses PyTorch 2.5.1 for CUDA 12.1 compatibility
- Includes all dependencies for the entire codebase
- GPU support included (CUDA required)