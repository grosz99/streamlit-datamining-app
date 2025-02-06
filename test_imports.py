import sys
print(f"Python path: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import sklearn
    print(f"sklearn version: {sklearn.__version__}")
except ImportError as e:
    print(f"Error importing sklearn: {e}")
