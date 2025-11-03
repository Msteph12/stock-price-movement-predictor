from pathlib import Path
import sys

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

print('Project root on sys.path:', project_root)

try:
    from src.dataload import load_stock_data
    from src.featureengineering import prepare_features
    print('Imports OK')

    df = load_stock_data('AAPL')
    print('Loaded df shape:', getattr(df, 'shape', 'no-shape'))
    print('Columns:', list(df.columns))
    print('Column dtypes:')
    print(df.dtypes)

    X, y = prepare_features(df)
    print('X shape:', getattr(X, 'shape', 'no-shape'), 'y shape:', getattr(y, 'shape', 'no-shape'))
    print('Feature columns sample:', list(X.columns)[:10])
except Exception as e:
    print('ERROR during smoke test:', type(e).__name__, e)
    raise
