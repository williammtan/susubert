from kfp.components import OutputPath, InputPath

def preprocess(input_path: InputPath(str), output_path: OutputPath(str)):
    import pandas as pd
    products = pd.read_csv(input_path)
    products = products.dropna(subset=['id', 'name', 'description'])
    products = products.dropna(subset=['master_product'])
    products.to_csv(output_path, index=False)

def train_test_split(
    matches_path: InputPath('str'),
    train_path: OutputPath(str),
    test_path: OutputPath(str),
    test_split: float=0.2,
    ):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from pathlib import Path
    
    matches = pd.read_csv(matches_path)
    train, test = train_test_split(matches, test_size=test_split)
    
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
