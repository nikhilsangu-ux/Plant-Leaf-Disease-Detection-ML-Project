"""Quickly check whether project dependencies are importable."""
import sys
print("Python:", sys.version.split()[0])
for module in ("tensorflow", "tensorflow_datasets", "numpy", "PIL", "sklearn", "matplotlib", "streamlit"):
    try:
        imported = __import__(module)
        print(f"OK  {module}: {getattr(imported, '__version__', 'installed')}")
    except Exception as exc:
        print(f"ERR {module}: {type(exc).__name__}: {exc}")
