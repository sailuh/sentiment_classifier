import pandas as pd

def split_dataset(df, val_frac=0.1, test_frac=0.2, gap=None):
    """
    Chronogolical train/validate/test split of a dataset.

    Args:
        df (pd.DataFrame): The input DataFrame to split.
        val_frac (float): Fraction of the dataset to use for validation.
        test_frac (float): Fraction of the dataset to use for testing.
        gap (int, optional): Number of rows to skip between splits to avoid data leakage.

    Returns:
        tuple: Three DataFrames corresponding to the training, validation, and test sets.
    """

    if df['updated_at'].isna().any():
        raise ValueError("The 'updated_at' column contains missing timestamps.")
    if not 0 < val_frac + test_frac < 1:
        raise ValueError("The sum of val_frac and test_frac must be between 0 and 1.")
    
    df = df.copy()
    df['updated_at'] = pd.to_datetime(df['updated_at'])
    if df['updated_at'].isna().any():
        raise ValueError("The 'updated_at' column contains invalid timestamps.")
    df = df.sort_values('updated_at', kind='stable').reset_index(drop=True)

    n = len(df)
    # Pick cutoffs from row positions, then split on the timestamp VALUE,
    # so rows sharing a timestamp can never straddle a boundary.
    val_start = df['updated_at'].iloc[int(n * (1 - val_frac - test_frac))]
    test_start = df['updated_at'].iloc[int(n * (1 - test_frac))]
    gap = gap or pd.Timedelta(0)

    t = df['updated_at']
    train = df[t < val_start - gap]
    val = df[(t >= val_start) & (t < test_start - gap)]
    test = df[t >= test_start]

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("A split is empty; too many ties or too little data")
    assert train['updated_at'].max() < val['updated_at'].min() <= val['updated_at'].max() < test['updated_at'].min()
    return train, val, test
