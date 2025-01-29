import pandas as pd

#Variabili Globali
train_data = pd.read_csv('train_auto.csv')
test_data = pd.read_csv('test_auto.csv')


#function for the main script
def load_data(train,test):
    try:
        train_data = pd.read_csv(train)
        test_data = pd.read_csv(test)
        return train_data, test_data
    except Exception as e:
        print(f'Error load data: {e}')
        raise


if __name__ == "__main__":
    # Train dataset overview
    print(train_data.info())
    print(train_data.describe())
    print(train_data.head(2))

    # Test dataset overview
    print(test_data.info())
    print(test_data.describe())
    print(test_data.head(2))
