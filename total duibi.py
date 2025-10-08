import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel("compare.xlsx", sheet_name="Sheet3")

# 提取列
y_true = df["Actual"].values
y_pred = df["Predicted"].values
hours = df["Hour"].values

# 计算 MAPE 和 RMSE
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print(f"MAPE: {mape:.2f}%")
print(f"RMSE: {rmse:.2f}")

# 绘制对比图
plt.figure(figsize=(10,6))
plt.plot(hours, y_pred, label="Predicted", marker="o")
plt.plot(hours, y_true, label="Actual", marker="s")

plt.title("Predicted vs Actual (Sheet3)", fontsize=14)
plt.xlabel("Hour", fontsize=12)
plt.ylabel("Load", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
