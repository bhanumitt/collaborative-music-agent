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

def create_env_file():
    """Create a sample .env file"""
    env_content = """# Collaborative Music Creation Agent - Environment Variables

# Spotify API (Optional - for extended functionality)
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here

# HuggingFace Settings
HF_TOKEN=your_huggingface_token_here

# Model Settings
MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
MAX_TOKENS=300
TEMPERATURE=0.7

# Demo Settings
DEMO_MODE=true
LOG_LEVEL=INFO

# Performance Settings
BATCH_SIZE=1
USE_QUANTIZATION=true
DEVICE=cpu
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    print("✓ Created .env.example file")

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

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
logs/
*.log
models/downloaded/
data/cache/
temp/
*.midi
*.wav
*.mp3

# Secrets
.env
config.json
secrets.json
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✓ Created .gitignore file")

def create_readme():
    """Create enhanced README.md"""
    readme_content = """# 🎵 Collaborative Music Creation Agent

An AI-powered music assistant that creates collaborative playlists and generates original music through conversational interaction.

## ✨ Features

- **🤖 Conversational AI**: Natural language interaction using Phi-3-mini LLM
- **🎼 Music Analysis**: Advanced audio feature extraction and analysis
- **🤝 Collaborative Playlists**: Find musical common ground between different tastes
- **🎵 Original Music Generation**: Create custom compositions blending multiple styles
- **📊 Explainable AI**: Understand why certain musical choices work well together
- **⚡ Real-time Performance**: Optimized for CPU inference with sub-second response times

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/collaborative-music-agent.git
   cd collaborative-music-agent
   ```

2. **Set up the environment**
   ```bash
   python setup.py
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser** to `http://localhost:7860`

## 🎯 Usage Examples

### Basic Conversation
```
You: "Create a playlist for me and my friends who love jazz and EDM"
Agent: "Great combination! To make the perfect blend, could you share 2-3 
        specific songs you each love? This helps me understand your exact vibe."
```

### Music Generation
```
You: "Generate an original song that blends classical and electronic styles"
Agent: "I'll create a unique composition that combines classical harmony with 
        electronic textures. Here's what I composed..."
```

## 🏗️ Architecture

### Core Components
- **LLM Interface**: Phi-3-mini for conversational AI
- **Music Analysis**: librosa + custom feature extraction  
- **Recommendation Engine**: Collaborative filtering + content-based
- **Music Generation**: music21 + rule-based composition
- **Web Interface**: Gradio for interactive demo

### Technology Stack
- **AI/ML**: PyTorch, Transformers, scikit-learn
- **Music**: librosa, music21, pretty_midi
- **Web**: Gradio, FastAPI
- **Data**: pandas, numpy

## 📊 Demo Data

The demo includes pre-analyzed songs across genres:
- Jazz classics (Miles Davis, John Coltrane, Dave Brubeck)
- Electronic hits (Deadmau5, Zedd, Avicii) 
- Rock anthems (Queen, Led Zeppelin, Eagles)
- Pop favorites (The Weeknd, Ed Sheeran)
- Fusion examples (Bonobo, Massive Attack)

## 🛠️ Development

### Project Structure
```
collaborative-music-agent/
├── src/
│   ├── llm_interface/          # Conversational AI
│   ├── music_analysis/         # Feature extraction
│   ├── recommendation/         # Playlist generation
│   ├── generation/            # Music creation
│   └── utils/                 # Helper functions
├── data/                      # Sample data
├── models/                    # AI models
├── tests/                     # Unit tests
├── app.py                     # Main application
└── requirements.txt           # Dependencies
```

### Running Tests
```bash
python -m pytest tests/
```

### Development Mode
```bash
python app.py --debug
```

## 🌟 Key Features in Detail

### Conversational Music Understanding
- Natural language processing for music preferences
- Context-aware follow-up questions
- Multi-turn conversation memory

### Advanced Music Analysis
- Audio feature extraction (tempo, energy, valence, etc.)
- Genre classification and compatibility scoring
- Musical similarity calculation

### Intelligent Playlist Generation
- Multi-objective optimization for group preferences
- Diversity-aware selection algorithms
- Smooth transitions between different styles

### Original Music Composition
- Rule-based composition with music theory
- Style fusion algorithms
- MIDI generation and export

## 🚀 Deployment

### HuggingFace Spaces
Ready for one-click deployment to HuggingFace Spaces:

1. Push code to GitHub
2. Connect to HuggingFace Spaces
3. Set to use `app.py` as main file
4. Deploy!

### Local Production
```bash
python app.py --host 0.0.0.0 --port 7860
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace**: For the amazing Transformers library and model hosting
- **Microsoft**: For the Phi-3-mini model
- **Music21**: For music theory and MIDI capabilities
- **librosa**: For audio analysis tools
- **Gradio**: For the intuitive web interface

## 📞 Support

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/collaborative-music-agent/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/collaborative-music-agent/discussions)

---

**🎵 Made with ❤️ for music lovers and AI enthusiasts**
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✓ Created enhanced README.md")

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
    
    # Create configuration files
    create_env_file()
    create_gitignore()
    create_readme()
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
    print("1. Copy .env.example to .env and configure your settings")
    print("2. Run 'python app.py' to start the application")
    print("3. Open http://localhost:7860 in your browser")
    print()
    print("🎵 Happy music making!")

if __name__ == "__main__":
    main()