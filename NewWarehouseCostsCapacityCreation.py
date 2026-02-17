import pandas as pd

#read .zip
zip_path = "CaseStudyDataPY.zip"
data_dir = "CaseStudyDataPY"

#read Excel with new warehouse data
file_name = 'Aggregated_Candidates_DemandWeighted.xlsx'
df = pd.read_excel(file_name)

#read Excel with Customer data
Customers_df = pd.read_csv(f"{data_dir}/Candidates.csv", index_col=0)
results = []

results = []

for _, row in df.iterrows():
    postcodes = [p.strip() for p in row['Original_Postcodes'].split(',')]

    subset = Customers_df[
        Customers_df['Postal District'].isin(postcodes)]
    results.append({
        'Candidate_ID': row['Candidate_ID'],
        'Total Setup Cost': subset['Setup cost'].sum(),
        'Operating Cost': subset['Operating'].sum(),
        'Total Capacity': subset['Capacity'].sum()})
df_aggregated = pd.DataFrame(results)
df_aggregated.to_excel("New_Warehouses_Costs_Capacity.xlsx", index=False)