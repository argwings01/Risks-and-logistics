import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# ==========================================
# 1. Custom Weighted K-Means
# ==========================================
def weighted_kmeans_custom(X, weights, n_clusters, max_iter=100, tol=1e-4, random_state=42):
    """
    A pure numpy implementation of Weighted K-Means.
    X: (n_samples, n_features) array
    weights: (n_samples,) array
    n_clusters: int
    """
    n_samples, n_features = X.shape
    
    # Randomly initialize centroids from the data points
    rng = np.random.RandomState(random_state)
    random_idx = rng.permutation(n_samples)[:n_clusters]
    centroids = X[random_idx]
    
    for iteration in range(max_iter):
        # --- E-step: Assign points to nearest centroid ---
        # Calculate distances (Squared Euclidean)
        # using (a-b)^2 = a^2 + b^2 - 2ab for efficiency
        # But for clarity/memory safety, we can just loop or use broadcasting
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            # dist = sum((X - center)^2)
            distances[:, k] = np.sum((X - centroids[k])**2, axis=1)
        
        labels = np.argmin(distances, axis=1)
        
        # --- M-step: Update centroids (Weighted Average) ---
        new_centroids = np.zeros((n_clusters, n_features))
        max_shift = 0.0
        
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                # Weighted average for this cluster
                cluster_points = X[mask]
                cluster_weights = weights[mask]
                total_w = np.sum(cluster_weights)
                
                if total_w > 0:
                    new_centroids[k] = np.average(cluster_points, axis=0, weights=cluster_weights)
                else:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If a cluster is empty, re-initialize it randomly to prevent it from dying
                # (Optional advanced step, simplified here to keep old centroid)
                new_centroids[k] = centroids[k]
        
        # Check convergence
        shift = np.sum((centroids - new_centroids)**2)
        if shift < tol:
            print(f"Converged at iteration {iteration}")
            break
        centroids = new_centroids
        
    return labels, centroids

# ==========================================
# 2. Data Parser (Same as before)
# ==========================================
def parse_case_study_data(file_path):
    print("Reading file...")
    with open(file_path, 'r') as f:
        content = f.read()

    def get_vector(key, dtype=float):
        pattern = re.compile(rf"{key}:\s*\[(.*?)\]", re.DOTALL)
        match = pattern.search(content)
        if match:
            clean_str = match.group(1).replace('"', '').replace('\n', ' ')
            values = clean_str.split()
            if dtype == float:
                return [float(x) for x in values]
            elif dtype == str:
                return [str(x) for x in values]
            elif dtype == int:
                return [int(x) for x in values]
        return []

    ids = get_vector("CustomerId", str)
    easting = get_vector("CustomerEasting", float)
    northing = get_vector("CustomerNorthing", float)

    # Parse Demands
    nb_customers = len(ids)
    total_demands = np.zeros(nb_customers)
    
    start_marker = "CustomerDemandPeriods: ["
    start_idx = content.find(start_marker)
    if start_idx != -1:
        pattern = re.compile(r"\((\d+)\s+(\d+)\s+(\d+)\)\s+([\d\s]+)")
        matches = pattern.finditer(content, start_idx)
        for match in matches:
            c_id = int(match.group(1)) - 1
            s_id = int(match.group(3))
            if s_id == 1: 
                demand_values = [float(x) for x in match.group(4).split()]
                total_demands[c_id] += sum(demand_values)

    df = pd.DataFrame({
        'Original_ID': ids,
        'Easting': easting,
        'Northing': northing,
        'Total_Demand': total_demands
    })
    
    # Handle zero demand
    df.loc[df['Total_Demand'] == 0, 'Total_Demand'] = 1.0
    return df

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    file_name = 'CaseStudyData.txt'
    
    try:
        # 1. Parse
        df_customers = parse_case_study_data(file_name)
        
        # 2. Run Custom K-Means
        print(f"\nRunning Weighted K-Means (Custom Implementation)...")
        X = df_customers[['Easting', 'Northing']].values
        weights = df_customers['Total_Demand'].values
        
        # Calling our custom function instead of sklearn
        labels, new_centroids = weighted_kmeans_custom(X, weights, n_clusters=50)
        
        df_customers['Cluster_ID'] = labels
        
        # 3. Organize Results
        agg_rows = []
        for c_id in range(50):
            # Centroids are already calculated by our custom function
            # But we need to sum up the demand for the report
            cluster_data = df_customers[df_customers['Cluster_ID'] == c_id]
            total_demand_cluster = cluster_data['Total_Demand'].sum()
            
            agg_rows.append({
                'Candidate_ID': c_id,
                'New_Easting': new_centroids[c_id][0],
                'New_Northing': new_centroids[c_id][1],
                'Aggregated_Demand_10Years': total_demand_cluster,
                'Num_Original_Nodes': len(cluster_data),
                'Original_Postcodes': ", ".join(cluster_data['Original_ID'].tolist())
            })
            
        df_candidates = pd.DataFrame(agg_rows)
        
        # 4. Save & Show
        df_candidates.to_csv('Aggregated_Candidates_DemandWeighted.csv', index=False)
        df_customers.to_csv('Customer_Cluster_Mapping_Demand.csv', index=False)
        
        print("\nSuccess! Files saved.")
        print(df_candidates[['Candidate_ID', 'New_Easting', 'New_Northing', 'Aggregated_Demand_10Years']].head())
        
        # Simple Plot
        plt.figure(figsize=(10, 12))
        plt.scatter(df_customers['Easting'], df_customers['Northing'], 
                   c=df_customers['Cluster_ID'], cmap='tab20', s=10, alpha=0.5)
        plt.scatter(df_candidates['New_Easting'], df_candidates['New_Northing'], 
                   c='red', marker='x', s=100)
        plt.title('Custom Weighted K-Means Result')
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Could not find {file_name}")
