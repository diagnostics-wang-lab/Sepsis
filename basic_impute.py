import pandas as pd
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # do linear interpolation to the first 6 columns (the vital signs)
    df.iloc[:, :6] = df.iloc[:, :6].interpolate()
    # fill in average for column 7 - 34
    df.iloc[:, 6:34] = df.iloc[:, 6:34].fillna(df.iloc[:, 6:34].mean())
    # fill still-missing front row values with succeeding value
    df = df.fillna(method='bfill', axis=0)
    # fill empty columns with 0s
    df = df.fillna(0)
    return df
    
# if __name__ == "__main__":
#     # load data
#     df = pd.read_csv("/Users/raina/desktop/sepsis_xg/training/p000001.psv", sep = "|")
#     df = impute_missing(df)
#     print(df)