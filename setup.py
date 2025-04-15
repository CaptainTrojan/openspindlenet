import os
import shutil
from setuptools import setup

# Create directory structure if it doesn't exist
os.makedirs("openspindlenet/models", exist_ok=True)
os.makedirs("openspindlenet/data", exist_ok=True)

# Copy model files if they exist in root models directory
root_models_dir = "models"
if os.path.exists(root_models_dir):
    for model_file in os.listdir(root_models_dir):
        if model_file.endswith(".onnx"):
            src_path = os.path.join(root_models_dir, model_file)
            dst_path = os.path.join("openspindlenet", "models", model_file)
            shutil.copy2(src_path, dst_path)
            print(f"Copied model file: {src_path} -> {dst_path}")

# Call setup() with no arguments - configuration is in pyproject.toml
setup()
