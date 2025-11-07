"""
Test script to verify core functionality of the UK Inflation Forecasting project
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def test_data_collection():
    """Test data collection module"""
    print("Testing data collection...")
    from data_collection import DataCollector
    
    collector = DataCollector(output_dir='data/raw')
    
    # Test inflation data collection
    inflation_df = collector.collect_inflation_data()
    assert len(inflation_df) > 0, "Inflation data should not be empty"
    assert 'cpi_inflation' in inflation_df.columns, "Should have cpi_inflation column"
    
    # Test economic indicators collection
    economic_df = collector.collect_economic_indicators()
    assert len(economic_df) > 0, "Economic data should not be empty"
    assert 'gdp_growth' in economic_df.columns, "Should have gdp_growth column"
    
    # Test merge
    merged_df = collector.merge_datasets(inflation_df, economic_df)
    assert len(merged_df) > 0, "Merged data should not be empty"
    
    print("✓ Data collection tests passed")
    return True


def test_data_preprocessing():
    """Test data preprocessing module"""
    print("Testing data preprocessing...")
    from data_preprocessing import DataPreprocessor
    import pandas as pd
    
    preprocessor = DataPreprocessor(input_dir='data/raw', output_dir='data/processed')
    
    # Load existing data
    df = preprocessor.load_data()
    assert len(df) > 0, "Data should load successfully"
    
    # Test lag features
    df_with_lags = preprocessor.create_lag_features(df)
    assert 'cpi_inflation_lag_1' in df_with_lags.columns, "Should create lag features"
    
    # Test rolling features
    df_with_rolling = preprocessor.create_rolling_features(df)
    assert 'cpi_inflation_rolling_mean_3' in df_with_rolling.columns, "Should create rolling features"
    
    # Test time features
    df_with_time = preprocessor.create_time_features(df)
    assert 'month' in df_with_time.columns, "Should create time features"
    
    print("✓ Data preprocessing tests passed")
    return True


def test_module_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from data_collection import DataCollector
        print("  ✓ data_collection imported")
    except Exception as e:
        print(f"  ✗ data_collection import failed: {e}")
        return False
    
    try:
        from data_preprocessing import DataPreprocessor
        print("  ✓ data_preprocessing imported")
    except Exception as e:
        print(f"  ✗ data_preprocessing import failed: {e}")
        return False
    
    try:
        from models import InflationForecaster
        print("  ✓ models imported (note: requires ML libraries)")
    except ImportError as e:
        print(f"  ⚠ models import requires ML libraries: {e}")
    
    try:
        from explainability import ModelExplainer
        print("  ✓ explainability imported (note: requires XAI libraries)")
    except ImportError as e:
        print(f"  ⚠ explainability import requires XAI libraries: {e}")
    
    try:
        from visualization import Visualizer
        print("  ✓ visualization imported (note: requires plotting libraries)")
    except ImportError as e:
        print(f"  ⚠ visualization import requires plotting libraries: {e}")
    
    print("✓ Core module imports successful")
    return True


def test_file_structure():
    """Test that all expected files exist"""
    print("Testing file structure...")
    
    expected_files = [
        'requirements.txt',
        'README.md',
        'run_pipeline.py',
        'quick_start.py',
        '.gitignore',
        'CONTRIBUTING.md',
        'src/__init__.py',
        'src/data_collection.py',
        'src/data_preprocessing.py',
        'src/models.py',
        'src/explainability.py',
        'src/visualization.py',
    ]
    
    expected_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'src',
        'notebooks',
        'models',
        'results',
        'results/plots',
        'results/reports',
    ]
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    missing_dirs = []
    for dir in expected_dirs:
        if not os.path.exists(dir):
            missing_dirs.append(dir)
    
    if missing_files:
        print(f"  ✗ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"  ✗ Missing directories: {missing_dirs}")
        return False
    
    print("✓ File structure tests passed")
    return True


def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("RUNNING TESTS FOR UK INFLATION FORECASTING PROJECT")
    print("="*70)
    print()
    
    results = []
    
    # Test file structure
    results.append(("File Structure", test_file_structure()))
    print()
    
    # Test module imports
    results.append(("Module Imports", test_module_imports()))
    print()
    
    # Test data collection
    try:
        results.append(("Data Collection", test_data_collection()))
    except Exception as e:
        print(f"✗ Data collection test failed: {e}")
        results.append(("Data Collection", False))
    print()
    
    # Test data preprocessing
    try:
        results.append(("Data Preprocessing", test_data_preprocessing()))
    except Exception as e:
        print(f"✗ Data preprocessing test failed: {e}")
        results.append(("Data Preprocessing", False))
    print()
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed successfully!")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
