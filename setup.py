#!/usr/bin/env python3
"""
Setup script for Collaborative Music Creation Agent
"""

import os
import subprocess
import sys
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure"""
    directories = [
        "src",
        "src/llm_interface",
        "src/music_analysis", 
        "src/recommendation",
        "src/generation",
        "src/utils",
        "data",
        "data/sample_songs",
        "data/features",
        "data/user_profiles",
        "models",
        "notebooks",
        "tests",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for Python packages"""
    init_files = [
        "src/__init__.py",
        "src/llm_interface/__init__.py",
        "src/music_analysis/__init__.py",
        "src/recommendation/__init__.py", 
        "src/generation/__init__.py",
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created: {init_file}")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    return True

def main():
    """Main setup function"""
    print("🎵 Setting up Collaborative Music Creation Agent...")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    print()
    
    # Create Python package files
    create_init_files()
    print()
    
    # Check if requirements.txt exists before installing
    if Path("requirements.txt").exists():
        install_deps = input("Install Python dependencies? (y/n): ").lower().strip()
        if install_deps in ['y', 'yes']:
            install_dependencies()
    else:
        print("⚠️  requirements.txt not found. Please create it first.")
    
    print()
    print("🎉 Setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Run 'python app.py' to start the application")
    print("2. Open http://localhost:7860 in your browser")
    print()
    print("🎵 Happy music making!")

if __name__ == "__main__":
    main()