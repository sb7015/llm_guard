# Llama Guard 3 Project

## Setup
1. Open Anaconda Prompt
2. Navigate to project: `cd C:\LlamaGuard`
3. Activate environment: `conda activate LlamaGuard`
4. Start Jupyter: `jupyter notebook`

## Usage
- Use `LlamaGuardDetector` class for content moderation
- Check `notebooks/llama_guard_testing.ipynb` for examples
- Select "LlamaGuard" kernel in Jupyter notebooks

## Files
- `environment.yml` - Conda environment definition
- `llama_guard_conda_setup.py` - Main setup script
- `notebooks/` - Jupyter notebooks
- `scripts/` - Python scripts
- `data/` - Data files

## Commands
```bash
# Activate environment
conda activate LlamaGuard

# Deactivate environment  
conda deactivate

# Update environment
conda env update -f environment.yml

# Remove environment
conda env remove -n LlamaGuard
```
