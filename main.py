import os
import subprocess

# ============================================================
# CLAIMWISE INSURANCE PREDICTION — MAIN ENTRY POINT
# ============================================================
# This is the main entry script for the ClaimWise Insurance
# Prediction project, following the Placement Prediction project structure.
# ============================================================

def main():
    print("=" * 70)
    print("       CLAIMWISE INSURANCE PREDICTION PROJECT       ")
    print("==================================================")
    print("\nProject Structure:")
    print(" - Dataset/              : CSV datasets and preprocessed splits")
    print(" - Src/                  : Preprocessing and EDA modules")
    print(" - Outputs/              : Visualization charts and correlation metrics")
    print(" - Models/               : Machine Learning model artifacts")
    print(" - General_Programs/     : Utility scripts")
    print(" - Static/ & templates/  : Web dashboard interface")
    print("=" * 70)

    pipeline_script = "claimwise_preprocessing_pipeline.py"
    if os.path.exists(pipeline_script):
        print(f"\nRunning main preprocessing pipeline: {pipeline_script}...\n")
        subprocess.run(["python3", pipeline_script])
    else:
        print(f"Error: {pipeline_script} not found.")

if __name__ == "__main__":
    main()
