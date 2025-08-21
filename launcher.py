"""
Simple launcher for Legal Document Review AI Agent.
This script avoids import issues during initial setup.
"""

import subprocess
import sys
import os
from pathlib import Path


def main():
    """Main launcher function."""
    print("🏛️ Legal Document Review AI Agent - Launcher")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        print(f"   Current version: {sys.version}")
        return
    
    print("✅ Python version check passed")
    
    # Create required directories
    required_dirs = ["data/uploads", "data/outputs", "data/temp", "logs"]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Directories created")
    
    # Check for .env file
    if not Path(".env").exists():
        print("⚠️  .env file not found. Copying from .env.example...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ .env file created")
        else:
            print("❌ .env.example file not found")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Please run manually: pip install -r requirements.txt")
        return
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next Steps:")
    print("1. Edit .env file and add your API keys:")
    print("   - OPENAI_API_KEY=your_openai_key_here")
    print("   - ANTHROPIC_API_KEY=your_anthropic_key_here")
    print("\n2. Choose how to run the agent:")
    print("   • Web UI: streamlit run src/ui/app.py")
    print("   • API Server: python -m src.api.main") 
    print("   • Example: python example.py")
    print("\n3. Or use the batch files:")
    print("   • start_ui.bat - Start the web interface")
    print("   • start_api.bat - Start the API server")
    
    print("\n⚠️  Important: You need either OpenAI or Anthropic API keys for the agent to work!")
    
    # Ask user what to do next
    print("\n" + "=" * 50)
    choice = input("What would you like to do now?\n1. Start Web UI\n2. Start API Server\n3. Run Example\n4. Exit\nChoice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Web UI...")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "src/ui/app.py"])
        except Exception as e:
            print(f"❌ Failed to start Web UI: {e}")
            print("   Try running manually: streamlit run src/ui/app.py")
    
    elif choice == "2":
        print("\n🚀 Starting API Server...")
        try:
            subprocess.run([sys.executable, "-m", "src.api.main"])
        except Exception as e:
            print(f"❌ Failed to start API Server: {e}")
            print("   Try running manually: python -m src.api.main")
    
    elif choice == "3":
        print("\n🚀 Running Example...")
        try:
            subprocess.run([sys.executable, "example.py"])
        except Exception as e:
            print(f"❌ Failed to run example: {e}")
            print("   Try running manually: python example.py")
    
    else:
        print("\n👋 Goodbye! Run this script again when you're ready to start.")


if __name__ == "__main__":
    main()
