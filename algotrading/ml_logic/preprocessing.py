import numpy as np
from sklearn.preprocessing import MinMaxScaler

def window(df, window_size=60):
    def split_df(df, start, end):
        split = df.iloc[start:end]
        X = split.iloc[:, :-1]
        y = split.iloc[:, -1]
        return X, y

    def window_data(X, y, window_size):
        X_windows = []
        y_windows = []
        for i in range(len(X) - window_size):
            X_windows.append(X[i:i + window_size])
            y_windows.append(y[i + window_size])
        return np.array(X_windows), np.array(y_windows)

    # Split the data into train, validation and test sets
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    X_train_df, y_train_df = split_df(df, 0, train_size)
    X_val_df, y_val_df = split_df(df, train_size, train_size + val_size)
    X_test_df, y_test_df = split_df(df, train_size + val_size, None)

    # Initialize the scaler
    sc = MinMaxScaler(feature_range=(0,1))

    # Fit the scaler on the training data and transform both training and testing data
    X_train = sc.fit_transform(X_train_df).astype('float')
    X_val = sc.transform(X_val_df).astype('float')
    X_test = sc.transform(X_test_df).astype('float')

    # Convert the target data to numpy arrays
    y_train = y_train_df.to_numpy().astype('float')
    y_val = y_val_df.to_numpy().astype('float')
    y_test = y_test_df.to_numpy().astype('float')

    # Create windows
    X_train_windows, y_train_windows = window_data(X_train, y_train, window_size)
    X_val_windows, y_val_windows = window_data(X_val, y_val, window_size)
    X_test_windows, y_test_windows = window_data(X_test, y_test, window_size)

    return X_train_windows, y_train_windows, X_val_windows, y_val_windows, X_test_windows, y_test_windows, sc
