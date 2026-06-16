import sys

libs = [
    ("pandas", "2.2.2"),
    ("numpy", "1.26.4"),
    ("sklearn", "1.5.1"),
    ("imblearn", "0.12.3"),
    ("xgboost", "2.1.1"),
    ("mlflow", "3.1.0"),
    ("shap", "0.46.0"),
    ("fastapi", "0.115.0"),
    ("pydantic", "2.8.2"),
    ("joblib", "1.4.2"),
]

print(f"Python: {sys.version.split()[0]}\n")
all_ok = True
for name, expected in libs:
    try:
        mod = __import__(name)
        actual = mod.__version__
        status = "✓" if actual == expected else "⚠ "
        if actual != expected:
            all_ok = False
        print(f"  {status}  {name:<20} {actual}  (expected {expected})")
    except ImportError:
        print(f"  ✗  {name:<20} NOT FOUND")
        all_ok = False

print()
if all_ok:
    print("✓ Environment is clean and ready.")
else:
    print("⚠  Some versions differ — this is usually fine unless there's a ✗.")
