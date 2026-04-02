import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# ======== 读取数据 ========
df = pd.read_csv("classification_results.csv")  # ← 修改为你的文件名

y_true = df.iloc[:,1]   # 第二列 = 真实标签
y_pred = df.iloc[:,2]   # 第三列 = 预测标签

# ======== 计算指标 ========
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1-score:", round(f1, 3))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# ======== 混淆矩阵可视化 ========
cm = confusion_matrix(y_true, y_pred)
labels = sorted(df.iloc[:,1].unique())  # 自动适应类别编号

plt.figure(figsize=(5,4), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("✅ 混淆矩阵已导出为 confusion_matrix.png")
