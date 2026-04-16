import subprocess
import sys
from pathlib import Path

def main():
    """
    Minimalist Runner for Modeling Pipeline.
    
    Responsibilities:
    1. Locate the python executable.
    2. Invoke 'main.py' with the 'run-monthly' command.
    
    Configuration (Horizon, Risk Threshold) is handled by 'main.py' 
    reading from 'config/job_config.json'. following Single Responsibility Principle.
    """
    python_exe = sys.executable
    script_path = Path(__file__).parent / "main.py"
    
    # Simple invocation. Arguments are optional in main.py thanks to config defaults.
    cmd = [
        python_exe, 
        str(script_path), 
        "run-monthly"
    ]
    
    print("-" * 60)
    print(f" Launching Modeling Pipeline...")
    print(f"   Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "-" * 60)
        print("Pipeline completed successfully.")
        print("-" * 60)
    except subprocess.CalledProcessError as e:
        print("\n" + "-" * 60)
        print(f"Pipeline failed with exit code {e.returncode}")
        print("-" * 60)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
