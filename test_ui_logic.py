type_mapping = {
    "不使用 (Unused)": "Unused",
    "ID (識別碼)": "ID",
    "數值特徵 (Numerical)": "Numerical",
    "類別特徵 (Categorical)": "Categorical",
    "時間特徵 (Datetime)": "Datetime",
    "預測目標 (Target)": "Target"
}
reverse_type_mapping = {v: k for k, v in type_mapping.items()}

print("Keys:", list(type_mapping.keys()))
print("Reverse mapping for ID:", reverse_type_mapping.get("ID"))
