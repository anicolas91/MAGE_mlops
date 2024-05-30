import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    
    # calculate duration and convert to minutes
    try:
        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    except:
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime

    df.duration = df.duration.dt.total_seconds() / 60

    # remove any outliers 
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # convert categorical location ids to strings
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df