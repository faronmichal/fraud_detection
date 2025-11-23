import pandas as pd
import numpy as np

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy to avoid modifying the original dataframe in memory
    df_processed = df.copy()
    
    # Check if 'Time' column exists (it might be missing in API requests if not provided, 
    # but it is always present in training data)
    if 'Time' in df_processed.columns:
        # Convert seconds to Hour of Day (0-24)
        df_processed['Hour_of_Day'] = (df_processed['Time'] % 86400) / 3600

        # Cyclic transformation (Sin/Cos)
        df_processed['Hour_Sin'] = np.sin(2 * np.pi * df_processed['Hour_of_Day'] / 24.0)
        df_processed['Hour_Cos'] = np.cos(2 * np.pi * df_processed['Hour_of_Day'] / 24.0)

        # Drop the raw 'Time' and 'Hour_of_Day' columns
        df_processed = df_processed.drop(columns=['Time', 'Hour_of_Day'])
        
    return df_processed