import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1.Завантажуємо оброблені дані
from data_loader import load_and_preprocess_data
X_train, y_train, X_test, passenger_ids = load_and_preprocess_data()

# 2. Дерево рішень
print("Навчаємо Дерево рішень...")
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 3. Робимо прогноз та створюємо файл для Kaggle
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('submission_decision_tree.csv', index=False)
print("Готово! Файл submission_decision_tree.csv успішно створено.")

# 4. Візуалізуємо дерево
print("Малюємо графік...")
plt.figure(figsize=(25,12), dpi=300) 
plot_tree(model, 
          feature_names=X_train.columns, 
          class_names=['Not Survived', 'Survived'], 
          filled=True, 
          rounded=True, 
          impurity=True, 
          proportion=False, 
          fontsize=10)
plt.title("Дерево рішень для прогнозування виживання на Титаніку", fontsize=20)
plt.savefig('decision_tree_final.png', bbox_inches='tight') 
print("Графік decision_tree_final.png збережено!")