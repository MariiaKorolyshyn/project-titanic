import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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


# 4. Візуалізація: вплив факторів на виживання
print("Малюємо графік...")
coefficients = model_lr.coef_[0]

plt.figure(figsize=(10, 6), dpi=300)
# Малюємо горизонтальні стовпчики
plt.barh(X_train.columns, coefficients, color='mediumpurple')
plt.title("Вплив факторів на виживання (Логістична регресія)", fontsize=16)
plt.xlabel("Вага фактора (більше 0 = плюс до виживання, менше 0 = мінус)")
plt.axvline(x=0, color='grey', linestyle='--') # Додаємо нульову лінію для наочності
plt.savefig('logistic_regression_importance.png', bbox_inches='tight')
print("Графік logistic_regression_importance.png збережено!")