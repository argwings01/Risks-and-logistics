import pandas as pd
import math

#read .zip
zip_path = "CaseStudyDataPY.zip"
data_dir = "CaseStudyDataPY"

#read Excel with new warehouse data
file_name = 'Aggregated_Candidates_DemandWeighted.xlsx'
df = pd.read_excel(file_name)

#Take coordinates of new warehouses
WarehouseCoordinates = df[['New_Easting', 'New_Northing']].values 

#read Excel with Supplier data
Customers_df = pd.read_csv(f"{data_dir}/Candidates.csv", index_col=0)
#Take coordinates of Customers (same locations as initial warehouses)
CustomerCoordinates = Customers_df[['X (Easting)', 'Y (Northing)']].values

#make matrix of distances from warehouses customers
Dist_Sup_to_Ware = [
    [0 for j in range(len(WarehouseCoordinates))]
    for i in range(len(CustomerCoordinates))]

for i in range(len(WarehouseCoordinates)):
    for j in range(len(CustomerCoordinates)):
        Dist_Sup_to_Ware[j][i]=math.sqrt(
            (CustomerCoordinates[j][0]-WarehouseCoordinates[i][0])**2 + 
            (CustomerCoordinates[j][1]-WarehouseCoordinates[i][1])**2)*192.61/228541

#192.61/228541 is an analogy of the distance in kilometers and the euclidian distance of the coordinates 
#of the distance between the first candidate location and the first supplier in the data. This gives us
#a way to transition coordinates to distance in kilometers

#write the matrix in an Excel
df_dist = pd.DataFrame(Dist_Sup_to_Ware)
df_dist.to_excel(
    "Distance_Warehouse_to_Customer.xlsx",
    index=False,
    header=False
)
