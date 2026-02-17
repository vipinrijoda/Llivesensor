import os
from sensor.constant.training_pipeline import SAVED_MODEL_DIR

print(f"SAVED_MODEL_DIR = {SAVED_MODEL_DIR}")
print(f"Absolute path: {os.path.abspath(SAVED_MODEL_DIR)}")

if os.path.exists(SAVED_MODEL_DIR):
    print("✅ Directory exists")
    print("Contents:")
    for item in os.listdir(SAVED_MODEL_DIR):
        print(f"  - {item}")
else:
    print("❌ Directory does NOT exist")