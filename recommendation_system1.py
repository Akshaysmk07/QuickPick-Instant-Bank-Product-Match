import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.tree import DecisionTreeClassifier
from functools import lru_cache
from joblib import Parallel, delayed
import time
import swifter  # For parallel operations
from multiprocessing import cpu_count

# Product name mapping
product_names = {
    "ind_ahor_fin_ult1": "Saving Account",
    "ind_aval_fin_ult1": "Guarantees",
    "ind_cco_fin_ult1": "Current Accounts",
    "ind_cder_fin_ult1": "Derivada Account",
    "ind_cno_fin_ult1": "Payroll Account",
    "ind_ctju_fin_ult1": "Junior Account",
    "ind_ctma_fin_ult1": "Más Particular Account",
    "ind_ctop_fin_ult1": "Particular Account",
    "ind_ctpp_fin_ult1": "Particular Plus Account",
    "ind_deco_fin_ult1": "Short-term Deposits",
    "ind_deme_fin_ult1": "Medium-term Deposits",
    "ind_dela_fin_ult1": "Long-term Deposits",
    "ind_ecue_fin_ult1": "E-account",
    "ind_fond_fin_ult1": "Funds",
    "ind_hip_fin_ult1": "Mortgage",
    "ind_plan_fin_ult1": "Plan Pensions",
    "ind_pres_fin_ult1": "Loans",
    "ind_reca_fin_ult1": "Taxes",
    "ind_tjcr_fin_ult1": "Credit Card",
    "ind_valo_fin_ult1": "Securities",
    "ind_viv_fin_ult1": "Home Account",
    "ind_nomina_ult1": "Payroll",
    "ind_nom_pens_ult1": "Pensions",
    "ind_recibo_ult1": "Direct Debit"
}



def load_csv(file_path='./df_train_small.csv.zip'):
    """Optimized CSV loading with parallel processing"""
    
    
    
    dtype_list = {
        'ind_cco_fin_ult1': 'uint8', 'ind_deme_fin_ult1': 'uint8',
        'ind_aval_fin_ult1': 'uint8', 'ind_valo_fin_ult1': 'uint8',
        'ind_reca_fin_ult1': 'uint8', 'ind_ctju_fin_ult1': 'uint8',
        'ind_cder_fin_ult1': 'uint8', 'ind_plan_fin_ult1': 'uint8',
        'ind_fond_fin_ult1': 'uint8', 'ind_hip_fin_ult1': 'uint8',
        'ind_pres_fin_ult1': 'uint8', 'ind_nomina_ult1': 'Int64', 
        'ind_cno_fin_ult1': 'uint8', 'ind_ctpp_fin_ult1': 'uint8',
        'ind_ahor_fin_ult1': 'uint8', 'ind_dela_fin_ult1': 'uint8',
        'ind_ecue_fin_ult1': 'uint8', 'ind_nom_pens_ult1': 'Int64',
        'ind_recibo_ult1': 'uint8', 'ind_deco_fin_ult1': 'uint8',
        'ind_tjcr_fin_ult1': 'uint8', 'ind_ctop_fin_ult1': 'uint8',
        'ind_viv_fin_ult1': 'uint8', 'ind_ctma_fin_ult1': 'uint8',
        'ncodpers': 'uint32'
    }
    
    name_col = ['ncodpers', 'fecha_dato', 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1',
               'ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
               'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1',
               'ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1',
               'ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
       
    start_time = time.time()
    
    # 1. Use larger chunks with parallel processing
    n_cores = cpu_count()
    chunk_size = max(int(5e6), 1000000)  # Larger chunks
    
    # 2. Use the 'c' engine which is faster for simple CSV parsing
    chunks = []
    for chunk in pd.read_csv(file_path, 
                           chunksize=chunk_size,
                           dtype=dtype_list, 
                           usecols=name_col,
                           compression='infer',  # Auto-detect compression
                           engine='c',
                           low_memory=False):  # Faster but uses more memory
        
        # Pre-filter using vectorized operations
        filtered_chunk = chunk[chunk.fecha_dato == '2015-05-28']
        if filtered_chunk.shape[0] > 0:
            chunks.append(filtered_chunk)
    
    if chunks:
        df_train1505 = pd.concat(chunks, ignore_index=True)
        # Process in one go after loading
        df_train1505 = df_train1505.drop(['fecha_dato'], axis=1)
        df_train1505 = df_train1505.fillna(0)
    else:
        df_train1505 = pd.DataFrame(columns=name_col)
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    
    return df_train1505

def add_user_input(arr, df):
    """Add user input to the dataframe with ID 0"""
    # Ensure correct structure: ncodpers should be first column
    if 'ncodpers' in df.columns and df.columns[0] != 'ncodpers':
        col_order = ['ncodpers'] + [col for col in df.columns if col != 'ncodpers']
        df = df[col_order]
    
    # Create array with ncodpers=0 and user selections
    arr = [0] + arr if len(arr) == len(df.columns) - 1 else [0] + list(arr)
    
    # Create user input dataframe
    user_input = pd.DataFrame([arr], columns=df.columns)
    
    # Add to existing dataframe
    result = pd.concat([user_input, df]).reset_index(drop=True)
    
    return result

def df_useritem(df, max_users=10000):
    """Process dataframe for user-item calculations with sample size control"""
    df_ui = df.copy()
    
    # Sample if needed to control memory usage
    if df_ui.shape[0] > max_users + 1:  # +1 to account for user with ID 0
        # Always keep the first row (user with ID 0)
        user_row = df_ui[df_ui.ncodpers == 0]
        remaining = df_ui[df_ui.ncodpers != 0].sample(max_users)
        df_ui = pd.concat([user_row, remaining])
    
    # Set index for faster access
    df_ui = df_ui.set_index('ncodpers')
    
    return df_ui

def cos_sim(df):
    """Calculate cosine similarity matrix optimized for memory usage"""
    # Use float32 to reduce memory usage
    start_time = time.time()
    cosine_sim = 1 - pairwise_distances(df.values.astype(np.float32), metric="cosine")
    
    sim_time = time.time() - start_time
    print(f"Similarity matrix calculated in {sim_time:.2f} seconds")
    
    return cosine_sim

def useritem(user_id, df, sim_matrix, k=20, sim_min_start=0.79, sim_min_floor=0.65, sim_step=0.025):
    """Calculate recommendations based on similar users - optimized implementation"""
    # Get the index of user_id in the similarity matrix
    try:
        cos_id = list(df.index).index(user_id)
    except ValueError:
        print(f"User ID {user_id} not found in dataframe")
        return {col: 0.0 for col in df.columns}
    
    # Initialize variables
    k_found = 0
    sim_min = sim_min_start
    user_sim_k = {}
    
    # Find similar users
    while k_found < k and sim_min >= sim_min_floor:
        for user_idx in range(len(df)):
            sim_score = sim_matrix[cos_id, user_idx]
            
            # Check if this user is similar enough but not the same
            if sim_min < sim_score < 0.99:
                user_sim_k[user_idx] = sim_score
                k_found += 1
                
                # Break early if we found enough users
                if k_found >= k:
                    break
        
        # Reduce similarity threshold if needed
        sim_min -= sim_step
    
    # Sort by similarity (descending)
    user_sim_k = dict(sorted(user_sim_k.items(), key=lambda item: item[1], reverse=True))
    user_id_k = list(user_sim_k.keys())
    
    # If no similar users found, return zeros
    if not user_id_k:
        return {col: 0.0 for col in df.columns}
    
    # Get dataframe with only similar users
    df_user_k = df.iloc[user_id_k]
    df_user_k_T = df_user_k.T
    
    # Rename columns to match user indices
    df_user_k_T.columns = user_id_k
    
    # Calculate mean ownership for each product
    result = {}
    for row_name, row in df_user_k_T.iterrows():
        result[row_name] = np.nanmean(row.values)
    
    # Handle case with all NaN values
    for key, value in result.items():
        if np.isnan(value):
            result[key] = 0.0
            
    return result

def df_mb(df):
    """Prepare dataframe for model-based recommendations"""
    df_mb = df.copy()
    df_mb = df_mb.set_index('ncodpers')
    return df_mb

def modelbased(user_id, df, model=None, n_jobs=4):
    """Calculate model-based recommendations in parallel"""
    if model is None:
        model = DecisionTreeClassifier(max_depth=9)
    
    # Get all product columns
    product_cols = df.columns
    
    # Train models in parallel
    def train_predict_product(col):
        """Train model for one product and predict for user"""
        # Split features and target
        y_train = df[col].astype('int')
        X_train = df.drop([col], axis=1)
        
        # Train model
        product_model = DecisionTreeClassifier(max_depth=9)
        product_model.fit(X_train, y_train)
        
        # Predict for target user
        user_features = X_train.loc[[user_id]]
        try:
            prob = product_model.predict_proba(user_features)[:, 1][0]
        except (IndexError, ValueError):
            prob = 0.0
            
        return col, prob
    
    # Use parallel processing
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_predict_product)(col) for col in product_cols
    )
    
    model_time = time.time() - start_time
    print(f"Model-based recommendations calculated in {model_time:.2f} seconds")
    
    return dict(results)

def popularity_based(df):
    """Calculate popularity-based recommendations"""
    start_time = time.time()
    
    # Get all product columns (excluding ncodpers)
    product_cols = [col for col in df.columns if col != 'ncodpers']
    
    # Calculate product popularities
    top_col = {}
    total_rows = df.shape[0]
    
    for col in product_cols:
        value_counts = df[col].value_counts()
        if 1 in value_counts:
            top_col[col] = np.around(value_counts[1] / total_rows, decimals=4)
        else:
            top_col[col] = 0.0
    
    pop_time = time.time() - start_time
    print(f"Popularity scores calculated in {pop_time:.2f} seconds")
    
    return top_col

def hybrid(user_id, df_p, df_u, sim_matrix, df_m, f1=0.5, f2=0.25, f3=0.25):
    """Calculate weighted hybrid recommendations"""
    start_time = time.time()
    
    # Get recommendations from each approach
    pb_h = popularity_based(df_p)
    ui_h = useritem(user_id, df_u, sim_matrix)
    mb_h = modelbased(user_id, df_m)
    
    # Combine with weights
    hybrid_scores = {}
    for k in pb_h.keys():
        if k in ui_h and k in mb_h:
            hybrid_scores[k] = (pb_h[k] * f1) + (ui_h[k] * f2) + (mb_h[k] * f3)
    
    hybrid_time = time.time() - start_time
    print(f"Hybrid recommendations calculated in {hybrid_time:.2f} seconds")
    
    return hybrid_scores

def change_names(col_names, map_products=product_names):
    """Convert technical column names to readable product names"""
    return [map_products.get(col_name, col_name) for col_name in col_names]

def recommendation(user_id, df, hybrid_outcome, top_n=7):
    """Generate final recommendations for user"""
    start_time = time.time()
    
    # Identify products user already owns
    if isinstance(df.index, pd.MultiIndex) or user_id not in df.index:
        user_row = df[df.index == user_id]
    else:
        user_row = df.loc[[user_id]]
    
    if user_row.empty:
        print(f"Warning: User {user_id} not found in dataframe")
        user_products = []
    else:
        user_products = [col for col in df.columns if user_row[col].iloc[0] == 1]
    
    # Filter out products user already owns
    recom = {k: v for k, v in hybrid_outcome.items() if k not in user_products}
    
    # Sort by score (descending)
    recom_sort = dict(sorted(recom.items(), key=lambda item: item[1], reverse=True))
    
    # Get top N recommendations
    top_products = list(recom_sort.keys())[:top_n]
    
    # Convert to readable names
    readable_recommendations = change_names(top_products)
    
    rec_time = time.time() - start_time
    print(f"Final recommendations generated in {rec_time:.2f} seconds")
    
    return readable_recommendations