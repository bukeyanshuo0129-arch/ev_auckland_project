import pandas as pd
import numpy as np

# 读取真实值和预测值
true_df = pd.read_excel("compare.xlsx", sheet_name="Sheet2")
pred_df = pd.read_excel("compare.xlsx", sheet_name="Sheet1")

# 合并数据 (按 Hour 对齐)
merged = pd.merge(true_df, pred_df, on="Hour", suffixes=("_true", "_pred"))

mape_results = {}
for col in [c for c in merged.columns if "_true" in c]:
    station = col.replace("_true", "")
    y_true = merged[f"{station}_true"].values
    y_pred = merged[f"{station}_pred"].values
    
    # 计算 MAPE（加1e-8避免除零错误）
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    mape_results[station] = mape

result = pd.DataFrame.from_dict(mape_results, orient="index", columns=["MAPE (%)"])
print(result)
result.to_excel("mape_results.xlsx")
