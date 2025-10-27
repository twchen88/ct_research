import pandas as pd

if __name__ == "__main__":
    train_csv = "./train.csv"  # path to dataset CSV file
    test_csv = "./test.csv"    # path to test dataset CSV file
    with open(train_csv) as df_file:
        data_source_df_train = pd.read_csv(df_file)
    
    with open(test_csv) as df_file:
        data_source_df_test = pd.read_csv(df_file)

    data_source_df = pd.concat([data_source_df_train, data_source_df_test], ignore_index=True)

    print("Total records loaded:", len(data_source_df))
    print("Missing values in each column:")
    print(data_source_df.isna().sum())

    # drop rows with missing values
    data_source_df = data_source_df.dropna()
    print("Total records after dropping missing values:", len(data_source_df))

    # Save the processed data to a new CSV file
    PROCESSED_DATA_CSV = "./processed_train.csv"
    data_source_df.to_csv(PROCESSED_DATA_CSV, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_CSV}")