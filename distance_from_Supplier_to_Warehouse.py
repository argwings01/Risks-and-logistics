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
Suppliers_df = pd.read_csv(f"{data_dir}/Suppliers.csv", index_col=0)
#Take coordinates of Suppliers
SupplierCoordinates = Suppliers_df[['X (Easting)', 'Y (Northing)']].values

#make matrix of distances from suppliers to warehouses
Dist_Sup_to_Ware = [
    [0 for j in range(len(WarehouseCoordinates))]
    for i in range(len(SupplierCoordinates))]

for i in range(len(SupplierCoordinates)):
    for j in range(len(WarehouseCoordinates)):
        Dist_Sup_to_Ware[i][j]=math.sqrt(
            (SupplierCoordinates[i][0]-WarehouseCoordinates[j][0])**2 + 
            (SupplierCoordinates[i][1]-WarehouseCoordinates[j][1])**2)*192.61/228541

#192.61/228541 is an analogy of the distance in kilometers and the euclidian distance of the coordinates 
#of the distance between the first candidate location and the first supplier in the data. This gives us
#a way to transition coordinates to distance in kilometers

#write the matrix in an Excel
df_dist = pd.DataFrame(Dist_Sup_to_Ware)
df_dist.to_excel(
    "Distance_Supplier_to_Warehouse.xlsx",
    index=False,
    header=False
)
