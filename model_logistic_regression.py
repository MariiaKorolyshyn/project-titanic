import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1. Завантажуємо оброблені дані
from data_loader import load_and_preprocess_data
X_train, y_train, X_test, passenger_ids = load_and_preprocess_data()

# 2. Навчаємо Логістичну регресію
print("Навчаємо Логістичну регресію...")
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)

# 3. Робимо прогноз та зберігаємо файл
predictions = model_lr.predict(X_test)
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_logistic_regression.csv', index=False)

print("Готово! Файл submission_logistic_regression.csv успішно створено.")