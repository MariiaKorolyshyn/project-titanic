import pandas as pd

def load_and_preprocess_data():
    print("Починаємо завантаження даних...")
    
    # Файли train.csv та test.csv мають лежати в тій самій папці, що й цей скрипт
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # Твоя функція обробки даних
    def preprocess(df):
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
        features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']
        return df[features]

    # Застосовуємо обробку
    X_train = preprocess(train)
    y_train = train['Survived']
    X_test = preprocess(test)
    
    # ID пасажирів потрібні для фінального файлу Kaggle
    passenger_ids = test['PassengerId'] 

    print("Дані успішно оброблено та готові до використання моделями!")
    
    return X_train, y_train, X_test, passenger_ids