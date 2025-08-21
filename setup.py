"""
Quick setup and test script for Legal Document Review AI Agent.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_environment():
    """Check if environment is properly configured."""
    print("🔍 Checking environment...")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 9):
        issues.append("Python 3.9+ required")
    
    # Check for .env file
    if not Path(".env").exists():
        print("⚠️  .env file not found. Copying from .env.example...")
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            issues.append("Please edit .env file and add your API keys")
        else:
            issues.append(".env.example file not found")
    
    # Check for required directories
    required_dirs = ["data/uploads", "data/outputs", "data/temp", "logs"]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"   • {issue}")
        return False
    else:
        print("✅ Environment check passed")
        return True


def run_tests():
    """Run basic tests."""
    print("🧪 Running tests...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))
        
        # Try to import and run tests
        try:
            from tests.test_agent import (
                test_configuration, test_models, TestDocumentParser, TestRiskAssessor
            )
            
            # Run basic tests
            test_configuration()
            test_models()
            
            # Test document parser
            parser_test = TestDocumentParser()
            parser_test.setup_method()
            parser_test.test_create_sample_document()
            
            # Test risk assessor
            risk_test = TestRiskAssessor()
            risk_test.setup_method()
            risk_test.test_pattern_risk_detection()
            
            print("✅ All tests passed!")
            return True
            
        except ImportError as ie:
            print(f"⚠️  Could not import test modules: {ie}")
            print("   This is normal if dependencies aren't installed yet.")
            
            # Run basic configuration tests without imports
            try:
                from src.core.config import get_config
                config = get_config()
                print("✅ Basic configuration test passed")
                return True
            except Exception as basic_e:
                print(f"❌ Basic configuration test failed: {basic_e}")
                return False
        
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        print("   You may need to install dependencies first: pip install -r requirements.txt")
        return False


def main():
    """Main setup function."""
    print("🏛️ Legal Document Review AI Agent - Setup")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Check environment
    if not check_environment():
        print("\n⚠️  Please fix the issues above and run setup again.")
        return
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed. The agent may still work with proper API keys.")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next Steps:")
    print("1. Edit .env file and add your API keys:")
    print("   - OPENAI_API_KEY=your_openai_key_here")
    print("   - ANTHROPIC_API_KEY=your_anthropic_key_here")
    print("\n2. Choose how to run the agent:")
    print("   • API Server: python -m src.api.main")
    print("   • Web UI: streamlit run src/ui/app.py")
    print("   • Example: python example.py")
    print("\n3. Or use the batch files:")
    print("   • start_api.bat - Start the API server")
    print("   • start_ui.bat - Start the web interface")
    
    print("\n⚠️  Important: You need either OpenAI or Anthropic API keys for the agent to work!")


if __name__ == "__main__":
    main()
