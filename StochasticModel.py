import pandas as pd
import xpress as xp
import numpy as np
import os
import zipfile
import sys


xp.init('C:/xpressmp//bin/xpauth.xpr')

zip_path = "CaseStudyDataPY.zip"
data_dir = "CaseStudyDataPY"

# Vehicle capacity in tonnes
VehicleCapacity = {
    1: 9.0,
    2: 2.4,
    3: 1.5
}

# Cost in pounds per mile travelled (fixed cost)
VehicleCostPerMileOverall = {
    1: 1.666,
    2: 1.727,
    3: 1.285
}

# Cost in pounds per mile and tonne transported (variable cost)
VehicleCostPerMileAndTonneOverall = {
    1: 0.185,
    2: 0.720,
    3: 0.857
}

# CO₂ emissions in kg per mile and tonne transported
VehicleCO2PerMileAndTonne = {
    1: 0.11,
    2: 0.31,
    3: 0.30
}



#==============================================================================
#Distance from Supplier to Warehouse matrix creation
file_name = 'Sliced_Candidate_Supplier_Distance.csv'
df1 = pd.read_csv(file_name)
Dist_Sup_to_Warehouse = df1.values

#Distance from Warehouse to Customer matrix creation
file_name = 'Sliced_Candidate_Customer_Distance.csv'
df2 = pd.read_csv(file_name)
Dist_Warehouse_to_Cust = df2.values

#Create vectors for setup costs, operation costs, and total capacity
file_name = 'New_Warehouses_Costs_Capacity.xlsx'
df3 = pd.read_excel(file_name)
TotalSetupCost = df3['Total Setup Cost'].values 
OperatingCost = df3['Operating Cost'].values 
TotalCapacity = df3['Total Capacity'].values

#demand matrix creation with rows for products and columns for customers
file_name = 'Demand.csv'
df4 = pd.read_csv(file_name)
demand_matrix_df = df4.pivot(index='Product', columns='Customer', values='Demand')
demand_matrix_df = demand_matrix_df.fillna(0)
demand_matrix = demand_matrix_df.values

#capacity matrix creation with rows for products, and columns for supplier
file_name = 'Suppliers.csv'
df5 = pd.read_csv(file_name)
supplier_cap_matrix_df = df5.pivot(index='ID', columns='Product group', values='Capacity')
supplier_cap_matrix_df = supplier_cap_matrix_df.fillna(0)
#create demand matrix with rows for products and columns for customers
supplier_cap_matrix = supplier_cap_matrix_df.values

#vehicle of each supplier
file_name = 'Suppliers.csv'
df6 = pd.read_csv(file_name)
Vehicle = df6['Vehicle type'].values 

#Product of each supplier
Product_Sup = df6['Product group'].astype(int).tolist()


#Create matrices with the demand of every customer 
#for each product in every period
demand_matrices = {}
for t in range(1, 11):
    file_name = f"Demand_Period_{t}.xlsx"
    df8 = pd.read_excel(file_name, header=None)
    demand_matrices[t] = df8.values


file_name = 'Aggregated_Customers_100.csv'
df7 = pd.read_csv(file_name)
Demand_P1_Total = df7['Demand_P1_Total'].values 
Demand_P2_Total = df7['Demand_P2_Total'].values 
Demand_P3_Total = df7['Demand_P3_Total'].values 
Demand_P4_Total = df7['Demand_P4_Total'].values 

Groups = range(df1.shape[0])
Supplier = range(df1.shape[1])
Customer = range(df8.shape[1])
Period = range(10)
Products = range(demand_matrix.shape[0])
Scenarios = range(20)

df9 = pd.read_csv("Stochastic_demand_100.csv")
#Sizes of each category
I = df9["Customer"].max() + 1
Q = df9["Product"].max() + 1
T = df9["Period"].max() + 1
S = df9["Scenario"].max() + 1

#Create vector with all the values for the demand in each scenario
DemandScenarios = np.zeros((I, Q, T, S))
for _, row in df9.iterrows():
    i = int(row["Customer"])
    q = int(row["Product"])
    t = int(row["Period"])
    s = int(row["Scenario"])
    DemandScenarios[i, q, t, s] = row["Demand"]


#Scenario Probabilities
p=[1/len(Scenarios)]*len(Scenarios)

# =============================================================================
# Transport cost calculations
# =============================================================================

CostSupplierWarehouse=Dist_Sup_to_Warehouse*0
for i in range(CostSupplierWarehouse.shape[0]):
    for j in range(CostSupplierWarehouse.shape[1]):
        if Vehicle[i]==1:
            CostSupplierWarehouse[i,j]=2*Dist_Sup_to_Warehouse[i,j]*VehicleCostPerMileAndTonneOverall[1]/1000
        elif Vehicle[i]==2:
            CostSupplierWarehouse[i,j]=2*Dist_Sup_to_Warehouse[i,j]*VehicleCostPerMileAndTonneOverall[2]/1000
            
CostWarehouseCustomer=Dist_Warehouse_to_Cust*0
for i in range(CostWarehouseCustomer.shape[0]):
    for j in range(CostWarehouseCustomer.shape[1]):
        CostWarehouseCustomer[i,j]=2*Dist_Warehouse_to_Cust[i,j]*VehicleCostPerMileAndTonneOverall[3]/1000

# =============================================================================
# Build optimization model
# =============================================================================
prob = xp.problem("Assignment 1")

# Parameters
y = prob.addVariables(Groups, Period, name='y', vartype=xp.binary)
z = prob.addVariables(Groups, Period, name='z', vartype=xp.binary)
x = prob.addVariables(Customer, Groups, Period, Scenarios, name='x', vartype=xp.binary)
M = prob.addVariables(Supplier, Groups, Period, Products, Scenarios, vartype=xp.integer, name="M")

obj_eqn = xp.Sum(z[j, t] * TotalSetupCost[j] +
       y[j, t] * OperatingCost[j] + xp.Sum(p[s]*xp.Sum(
      xp.Sum(CostSupplierWarehouse[(j,k)] * M[k, j, t, q,s] for k in Supplier) +
      xp.Sum(CostWarehouseCustomer[(j,i)] * DemandScenarios[i, q, t, s] * x[i, j, t,s] for i in Customer)  for q in Products)
      for s in Scenarios) for j in Groups for t in Period)


prob.setObjective(obj_eqn , sense=xp.minimize)


# Constraints
prob.addConstraint(y[j, t] >= y[j, t-1] for j in Groups for t in Period if t > 0)


prob.addConstraint(x[i, j, t,s] <= y[j, t] for i in Customer for j in Groups for t in Period for s in Scenarios)


prob.addConstraint(xp.Sum(z[j, t] for t in Period) <= 1   for j in Groups)


prob.addConstraint(xp.Sum(x[i, j, t,s] for j in Groups) == 1  for i in Customer for t in Period  for s in Scenarios)


prob.addConstraint(M[k,j,t,q,s] == 0 for k in Supplier for q in Products if q+1 != Product_Sup[k] for j in Groups for t in Period  for s in Scenarios)


prob.addConstraint(y[j,0]==z[j,0] for j in Groups)


prob.addConstraint(y[j,t]-y[j,t-1]==z[j,t] for j in Groups for t in Period if t > 0)


prob.addConstraint(
    y[j,t] * supplier_cap_matrix[k,q]>=M[k, j, t, q,s]>= 0
    for k in Supplier for j in Groups for t in Period for q in Products  for s in Scenarios)


prob.addConstraint(
    xp.Sum(DemandScenarios[i, q, t, s] * x[i,j,t,s] for i in Customer)
    <= xp.Sum(M[k,j,t,q,s] for k in Supplier)
    for j in Groups for q in Products for t in Period for s in Scenarios)


prob.addConstraint(
    xp.Sum(DemandScenarios[i, q, t, s] * x[i,j,t,s] for i in Customer for q in Products)<= 
    xp.Sum(M[k, j, t, q,s] for k in Supplier for q in Products) <= TotalCapacity[j] * y[j, t]
    for j in Groups for t in Period for s in Scenarios)

# To turn on and off the solver log
xp.setOutputEnabled(True)

prob.solve()



# =============================================================================
# Post-processing and data visualisation
# =============================================================================

sol_status = prob.attributes.solstatus

if sol_status == xp.SolStatus.OPTIMAL:
    print("Optimal solution found")
    best_obj = prob.attributes.objval
    best_bound = prob.attributes.bestbound
    mip_gap = abs(best_obj - best_bound) / (1e-10 +abs(best_obj))
    print(f"MIP Gap: {mip_gap*100:.2f}%")
    
elif sol_status == xp.SolStatus.FEASIBLE:
    print("Feasible solution (not proven optimal)")
    best_obj = prob.attributes.objval
    best_bound = prob.attributes.bestbound
    mip_gap = abs(best_obj - best_bound) / (1e-10 +abs(best_obj))
    print(f"MIP Gap: {mip_gap*100:.2f}%")
elif sol_status == xp.SolStatus.INFEASIBLE:
    print("Model is infeasible")
elif sol_status == xp.SolStatus.UNBOUNDED:
    print("Model is unbounded")
else:
    print("No solution available")
    
# for j in Groups:
#     for t in Period:
#         if prob.getSolution(z[j,t])>0:
#             print("z(",j,",",t,") is: ", prob.getSolution(z[j,t]))

# for j in Groups:
#     for t in Period:
#         if prob.getSolution(y[j,t])>0:
#             print("y(",j,",",t,") is: ", prob.getSolution(y[j,t]))
            
           
# for j in Groups:
#     for t in Period:
#         if prob.getSolution(y[j,t])>0:
#             for k in Supplier: 
#                 for q in Products:
#                     for s in Scenarios:
#                         if prob.getSolution(M[k,j,t,q,s])>0:
#                             print("M(",k,",",j,",",t,",",q,",",s,") is: ", prob.getSolution(M[k,j,t,q,s]))

# for j in Groups:
#     for t in Period:   
#         if prob.getSolution(y[j,t])>0:
#             for i in Customer:
#                 for s in Scenarios:
#                     if prob.getSolution(x[i,j,t,s])>0:
#                         print("x(",i,",",j,",",t,",",s,") is: ", prob.getSolution(x[i,j,t,s]))


opt_val = prob.getObjVal()
print("Optimal objective value:", opt_val)
