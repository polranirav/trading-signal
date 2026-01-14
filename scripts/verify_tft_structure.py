#!/usr/bin/env python3
"""
Verify TFT Implementation Structure.

Checks that all TFT files exist, have correct structure, and can be parsed.
This works without requiring dependencies to be installed.

Usage:
    python scripts/verify_tft_structure.py
"""

import sys
from pathlib import Path
import ast

print("=" * 80)
print("TFT IMPLEMENTATION STRUCTURE VERIFICATION")
print("=" * 80)
print()

errors = []
warnings = []

# Test 1: Check file existence
print("Test 1: Check file existence...")
required_files = [
    "src/analytics/tft.py",
    "tests/unit/test_tft.py",
    "scripts/train_tft.py",
    "src/analytics/__init__.py",
    "src/analytics/confluence.py",
]

for file_path in required_files:
    path = Path(file_path)
    if path.exists():
        size = path.stat().st_size
        lines = len(path.read_text().splitlines())
        print(f"   ✅ {file_path} exists ({lines} lines, {size:,} bytes)")
    else:
        print(f"   ❌ {file_path} NOT FOUND")
        errors.append(f"Missing file: {file_path}")

print()

# Test 2: Check syntax (parse Python files)
print("Test 2: Check Python syntax...")
for file_path in ["src/analytics/tft.py", "tests/unit/test_tft.py", "scripts/train_tft.py"]:
    path = Path(file_path)
    if path.exists():
        try:
            with open(path, 'r') as f:
                code = f.read()
            ast.parse(code)
            print(f"   ✅ {file_path} syntax valid")
        except SyntaxError as e:
            print(f"   ❌ {file_path} syntax error: {e}")
            errors.append(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            print(f"   ⚠️  {file_path} parse error: {e}")
            warnings.append(f"Parse error in {file_path}: {e}")

print()

# Test 3: Check for required classes
print("Test 3: Check for required classes in tft.py...")
if Path("src/analytics/tft.py").exists():
    with open("src/analytics/tft.py", 'r') as f:
        code = f.read()
    
    required_classes = [
        "TemporalFusionTransformer",
        "VariableSelectionNetwork",
        "TFTTrainer",
        "TFTPredictionResult",
    ]
    
    for class_name in required_classes:
        if f"class {class_name}" in code:
            print(f"   ✅ {class_name} class found")
        else:
            print(f"   ❌ {class_name} class NOT FOUND")
            errors.append(f"Missing class: {class_name}")

print()

# Test 4: Check for required methods
print("Test 4: Check for required methods in TFT class...")
if Path("src/analytics/tft.py").exists():
    with open("src/analytics/tft.py", 'r') as f:
        code = f.read()
    
    required_methods = [
        "forward",
        "predict_with_quantiles",
        "quantile_loss",
        "prepare_tft_sequences",
        "create_dataloaders",
        "train",
    ]
    
    for method_name in required_methods:
        if f"def {method_name}" in code:
            print(f"   ✅ {method_name} method found")
        else:
            print(f"   ⚠️  {method_name} method not found (might be in different class)")

print()

# Test 5: Check integration in ConfluenceEngine
print("Test 5: Check TFT integration in ConfluenceEngine...")
if Path("src/analytics/confluence.py").exists():
    with open("src/analytics/confluence.py", 'r') as f:
        code = f.read()
    
    integration_checks = [
        ("_get_tft_score", "TFT scoring method"),
        ("_get_ml_score", "ML scoring method"),
        ("TemporalFusionTransformer", "TFT import"),
        ("TFTPredictionResult", "TFT result import"),
    ]
    
    for check_str, description in integration_checks:
        if check_str in code:
            print(f"   ✅ {description} found in ConfluenceEngine")
        else:
            print(f"   ⚠️  {description} not found in ConfluenceEngine")
            warnings.append(f"{description} not found in confluence.py")

print()

# Test 6: Check module exports
print("Test 6: Check module exports in __init__.py...")
if Path("src/analytics/__init__.py").exists():
    with open("src/analytics/__init__.py", 'r') as f:
        code = f.read()
    
    export_checks = [
        "TemporalFusionTransformer",
        "TFTTrainer",
        "TFTPredictionResult",
    ]
    
    for export in export_checks:
        if export in code:
            print(f"   ✅ {export} exported")
        else:
            print(f"   ⚠️  {export} not exported")
            warnings.append(f"{export} not exported in __init__.py")

print()

# Test 7: Check test structure
print("Test 7: Check test structure...")
if Path("tests/unit/test_tft.py").exists():
    with open("tests/unit/test_tft.py", 'r') as f:
        code = f.read()
    
    test_checks = [
        ("class TestVariableSelectionNetwork", "Variable Selection Network tests"),
        ("class TestTemporalFusionTransformer", "TFT architecture tests"),
        ("class TestTFTTrainer", "TFTTrainer tests"),
        ("def test_", "Test functions (should have multiple)"),
    ]
    
    for check_str, description in test_checks:
        count = code.count(check_str)
        if count > 0:
            if check_str == "def test_":
                print(f"   ✅ {count} test functions found")
            else:
                print(f"   ✅ {description} found")
        else:
            print(f"   ⚠️  {description} not found")
            warnings.append(f"{description} not found in test_tft.py")

print()

# Test 8: Check training script structure
print("Test 8: Check training script structure...")
if Path("scripts/train_tft.py").exists():
    with open("scripts/train_tft.py", 'r') as f:
        code = f.read()
    
    script_checks = [
        ("def train_tft_model", "train_tft_model function"),
        ("def prepare_training_data", "prepare_training_data function"),
        ("def main", "main function"),
        ("argparse", "Argument parser"),
        ("--symbol", "Symbol argument"),
        ("--epochs", "Epochs argument"),
    ]
    
    for check_str, description in script_checks:
        if check_str in code:
            print(f"   ✅ {description} found")
        else:
            print(f"   ⚠️  {description} not found")
            warnings.append(f"{description} not found in train_tft.py")

print()

# Test 9: Check Celery tasks
print("Test 9: Check Celery tasks for TFT...")
if Path("src/tasks/ml_tasks.py").exists():
    with open("src/tasks/ml_tasks.py", 'r') as f:
        code = f.read()
    
    task_checks = [
        ("def train_tft", "train_tft task"),
        ("def train_tft_all", "train_tft_all task"),
        ("TemporalFusionTransformer", "TFT import in tasks"),
        ("TFTTrainer", "TFTTrainer import in tasks"),
    ]
    
    for check_str, description in task_checks:
        if check_str in code:
            print(f"   ✅ {description} found")
        else:
            print(f"   ⚠️  {description} not found")
            warnings.append(f"{description} not found in ml_tasks.py")

print()

# Summary
print("=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)
print()

if errors:
    print(f"❌ ERRORS FOUND: {len(errors)}")
    for error in errors:
        print(f"   - {error}")
    print()
else:
    print("✅ No critical errors found")
    print()

if warnings:
    print(f"⚠️  WARNINGS: {len(warnings)}")
    for warning in warnings:
        print(f"   - {warning}")
    print()

if not errors:
    print("✅ TFT Implementation Structure: VERIFIED")
    print()
    print("Next Steps:")
    print("  1. Install dependencies: pip install -e . (or your package manager)")
    print("  2. Run full tests: python scripts/test_tft_implementation.py")
    print("  3. Run unit tests: pytest tests/unit/test_tft.py -v")
    print("  4. Train TFT model: python scripts/train_tft.py --symbol AAPL")
    print()
    print("Note: PyTorch (torch) is required for actual execution.")
    print("      It's listed in pyproject.toml dependencies.")
    print()
else:
    print("❌ Structure verification failed. Please fix errors above.")
    print()

print("=" * 80)

sys.exit(0 if not errors else 1)
