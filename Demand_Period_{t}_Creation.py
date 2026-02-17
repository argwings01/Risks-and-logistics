import pandas as pd

df7 = pd.read_csv("Aggregated_Demand_Details.csv")
for t in sorted(df7["Period"].unique()):
    df_t = df7[df7["Period"] == t]
    demand_matrix = df_t.pivot(
        index="Product",
        columns="Customer",
        values="Demand")
    
    file_name = f"Demand_Period_{t}.xlsx"
    demand_matrix.to_excel(
        file_name,
        index=False,
        header=False)
    print(f"Saved {file_name}")
