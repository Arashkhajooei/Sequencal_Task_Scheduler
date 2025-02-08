import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from clickhouse_driver import Client
import pymysql
import time

# For clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Default arguments for the DAG
default_args = {
    'owner': 'Arash',
    'depends_on_past': False,
    'start_date': datetime(2025, 2, 5),
    'retries': 10,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'Incremental_Rides_Automation',
    catchup=False,
    default_args=default_args,
    description='DAG for incremental rides automation',
    schedule_interval='@daily'
)

# Load Database Credentials (Censored)
USER = '*****'
PASSWORD = '*****'
HOST = '*****'
DB_NAME = "********"

CLICKHOUSE_HOST = '*****'
CLICKHOUSE_PORT = '*****'
CLICKHOUSE_USER = '*****'
CLICKHOUSE_PASSWORD = '*****'

def get_mysql_engine(database=DB_NAME):
    """Returns a SQLAlchemy engine connected to the specified database."""
    return create_engine(f"mysql+pymysql://{USER}:{PASSWORD}@{HOST}/{database}")

# ClickHouse connection
client = Client(
    host=CLICKHOUSE_HOST,
    port=int(CLICKHOUSE_PORT),
    user=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASSWORD
)


########################################
# Chunked Writing Helpers
########################################
def calculate_chunks_needed(df, chunk_size):
    """
    Returns the number of chunks needed to split `df`
    so that each chunk has at most `chunk_size` rows.
    """
    if len(df) == 0:
        return 0
    return (len(df) - 1) // chunk_size + 1

def write_chunk_to_sql(chunk, table_name, engine, if_exists="append"):
    """
    Write a single chunk of data to a SQL table using pandas.to_sql().
    """
    chunk.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)

def write_to_sql_in_chunks(df, table_name, engine, chunk_size=100000):
    """
    Writes a DataFrame to SQL in multiple chunks.
    The first chunk uses if_exists='replace' by default,
    subsequent chunks use if_exists='append'.
    """
    num_chunks = calculate_chunks_needed(df, chunk_size)
    if num_chunks == 0:
        print(f"No data to write to {table_name}.")
        return
    
    print(f"Writing {table_name} in {num_chunks} chunks of up to {chunk_size} rows each...")

    chunks = np.array_split(df, num_chunks)
    for i, chunk in enumerate(chunks):
        try:
            if i == 0:
                # For the first chunk, replace the table if it exists
                write_chunk_to_sql(chunk, table_name, engine, if_exists="replace")
            else:
                # For subsequent chunks, append
                write_chunk_to_sql(chunk, table_name, engine, if_exists="append")
            print(f"Chunk {i+1}/{num_chunks} successfully written to {table_name}")
        except Exception as e:
            print(f"Failed to write chunk {i+1}/{num_chunks} to {table_name}: {e}")
            raise

########################################
# Main DAG Functions
########################################

def execute_query(start_date, finish_date):
    """
    Reads /opt/airflow/dags/fetch_all_customer_data.sql,
    formats it with start_date & finish_date,
    and executes against ClickHouse.
    """
    try:
        # Make sure your file is located here:
        with open('/opt/airflow/dags/fetch_all_customer_data.sql', 'r') as file:
            query = file.read().format(start_date=start_date, finish_date=finish_date)
        
        result, columns = client.execute(query, with_column_types=True, settings={'max_block_size': 100000})
        return pd.DataFrame(result, columns=[col[0] for col in columns])
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

def fetch_all_customer_data():
    """
    1. Reads data from ClickHouse in monthly increments (past ~56 days).
    2. Concatenates the data into a single DataFrame.
    3. Writes to MySQL table Sales_Clustering.Full_customers_data in chunks.
    """
    full_customers_data = pd.DataFrame()
    try:
        # 56 days in the past up to today
        current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=56)
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Loop month-by-month
        while current_date <= end_date:
            start_date = current_date.strftime('%Y-%m-%d')  # 'YYYY-MM-DD'
            # Advance to next month
            if current_date.month == 12:
                next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                next_month = current_date.replace(month=current_date.month + 1, day=1)
            
            # Ensure finish_date is also 'YYYY-MM-DD'
            finish_date_candidate = (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Compare as datetime objects
            finish_date_dt = datetime.strptime(finish_date_candidate, '%Y-%m-%d')
            end_date_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
            finish_date = min(finish_date_dt, end_date_dt).strftime('%Y-%m-%d')
            
            print(f"Executing ClickHouse query for {start_date} to {finish_date}...")
            data = execute_query(start_date, finish_date)
            if not data.empty:
                full_customers_data = pd.concat([full_customers_data, data], ignore_index=True)
                print(f"Data for {start_date} to {finish_date} appended successfully.")
            else:
                print(f"No data found for {start_date} to {finish_date}, skipping.")
            
            current_date = next_month
        
        print("All data has been collected. Now writing to MySQL in chunks...")
        print(full_customers_data.head())  # Show first few rows for debugging

        # Save final DataFrame to MySQL in chunks
        engine = get_mysql_engine(DB_NAME)
        write_to_sql_in_chunks(full_customers_data, "Full_customers_data", engine, chunk_size=100000)
        print("Data successfully written to Sales_Clustering.Full_customers_data.")

    except Exception as e:
        print(f"Error fetching customer data: {e}")
        raise e

def calculate_metrics_and_cluster():
    """
    Reads Full_customers_data from MySQL, calculates customer metrics including additional behavioral metrics,
    performs outlier filtering, scales the features, applies KMeans clustering,
    writes the detailed clustering results to Sales_Clustering.Clustered_Results,
    and writes a cluster summary (mean of numeric columns) to Sales_Clustering.Clusters_Summary.
    """
    # --- Read data from MySQL ---
    engine = get_mysql_engine(DB_NAME)
    query = "SELECT * FROM Sales_Clustering.Full_customers_data"
    dynamic_attributes = pd.DataFrame()
    for attempt in range(10):
        try:
            dynamic_attributes = pd.read_sql(query, engine)
            print("Read Full_customers_data successfully!")
            break
        except Exception as e:
            print(f"Error reading Full_customers_data (attempt {attempt + 1}/10): {str(e)}")
            time.sleep(5)
    else:
        raise Exception("Failed to read Full_customers_data after 10 attempts.")

    df = dynamic_attributes.copy()
    if df.empty:
        raise ValueError("Data read from MySQL is empty. Cannot proceed.")

    # --- Preprocessing ---
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
    df['date'] = df['created_date'].dt.date
    df['is_delivered'] = df['status'].isin(['DELIVERED']).astype(int)
    df['is_request'] = df['status'].isin(['DELIVERED', 'CANCELLED']).astype(int)

    # --- Aggregation: Basic Customer Metrics ---
    agg_df = df.groupby('customer_request_id').agg(
        first_order_date=('created_date', 'min'),
        last_order_date=('created_date', 'max'),
        delivered_orders=('is_delivered', 'sum'),
        total_requests=('is_request', 'sum'),
        request_days=('date', 'nunique')
    ).reset_index()

    reference_date = df['created_date'].max()
    agg_df['recency'] = (reference_date - agg_df['last_order_date']).dt.days

    # --- Period Calculations (Active Period) ---
    agg_df['period_days'] = (agg_df['last_order_date'] - agg_df['first_order_date']).dt.days + 1
    agg_df.loc[agg_df['period_days'] <= 0, 'period_days'] = 1
    agg_df['period_weeks'] = agg_df['period_days'] / 7.0

    # --- Basic Daily Metrics ---
    agg_df['avg_delivered_daily'] = agg_df['delivered_orders'] / agg_df['period_days']
    agg_df['avg_requests_daily'] = agg_df['total_requests'] / agg_df['period_days']
    agg_df['request_freq_daily'] = agg_df['request_days'] / agg_df['period_days']
    agg_df['delivered_freq_daily'] = agg_df['delivered_orders'] / agg_df['period_days']

    # --- Additional Behavioral Metrics ---
    # 1. Variance in delivered orders per day
    daily_delivered = df.groupby(['customer_request_id', 'date']).agg(
        daily_delivered=('is_delivered', 'sum')
    ).reset_index()
    variance_df = daily_delivered.groupby('customer_request_id')['daily_delivered'].var().reset_index()
    variance_df.rename(columns={'daily_delivered': 'variance_delivered_daily'}, inplace=True)
    variance_df['variance_delivered_daily'] = variance_df['variance_delivered_daily'].fillna(0)
    agg_df = pd.merge(agg_df, variance_df, on='customer_request_id', how='left')
    
    # 2. Average time between delivered orders
    agg_df['avg_time_between_delivered'] = np.where(
        agg_df['delivered_orders'] > 1,
        agg_df['period_days'] / (agg_df['delivered_orders'] - 1),
        agg_df['period_days']
    )

    # --- Define Clustering Features ---
    features = [
        "recency",                    # days since last order
        "period_days",                # active period (days)
        "delivered_orders",           # cumulative delivered orders
        "avg_delivered_daily",        # daily average delivered orders
        "variance_delivered_daily",   # variance in daily delivered orders
        "avg_time_between_delivered", # average time between delivered orders
        "avg_requests_daily"          # daily average requests
    ]
    X = agg_df[features].fillna(0)
    
    # --- Outlier Filtering using Z-Scores ---
    from scipy import stats
    z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
    threshold = 3
    outlier_mask = (z_scores > threshold).any(axis=1)
    print("Number of outlier records:", outlier_mask.sum())
    filtered_df = agg_df[~outlier_mask].copy()
    print("Number of records after outlier removal:", filtered_df.shape[0])
    
    # --- Scaling and Clustering ---

    scaler = StandardScaler()
    X_filtered = filtered_df[features].fillna(0)
    X_filtered_scaled = scaler.fit_transform(X_filtered)
    
    kmeans_final = KMeans(n_clusters=5, random_state=42, n_init=10)
    filtered_df["cluster"] = kmeans_final.fit_predict(X_filtered_scaled)
    print("Clustering complete! Here's a sample:")
    print(filtered_df.head())
    
    # --- Write Clustered Results ---
    filtered_df.to_sql("Clustered_Results", con=engine, if_exists="replace", index=False)
    # --- Compute and Write Cluster Summary ---
    # Use all numeric columns from filtered_df for the summary.
    numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns
    cluster_summary = filtered_df.groupby("cluster")[numeric_columns].mean()
    cluster_summary.to_sql("Clusters_Summary", con=engine, if_exists="replace", index=False)

    
    
def fetch_all_target_data():
    """Fetches target data from ClickHouse, processes it to compute metrics that match the clustering features,
    and assigns each customer to the closest cluster based on Euclidean distance.
    The final processed data is written to Sales_Clustering.Final_Target_Data_Clusters.
    """
    # --- Step 1: Get Customer Data from MySQL ---
    engine = get_mysql_engine("Sales")
    query = "SELECT DISTINCT customer_id, joining_date FROM Sales.Locked_List ORDER BY joining_date DESC"
    customer_df = pd.read_sql(query, engine)
    customer_df['joining_date'] = pd.to_datetime(customer_df['joining_date'])
    customer_ids = customer_df['customer_id'].tolist()
    customer_mask = ','.join([f"'{customer}'" for customer in customer_ids])
    
    # --- Step 2: Fetch Data from ClickHouse ---
    with open('/opt/airflow/dags/fetch_all_target_data.sql', 'r') as file:
        sql_template = file.read()
    query = sql_template.format(
        start_date="2024-03-01",
        finish_date=datetime.today().strftime('%Y-%m-%d'),
        customer_mask=customer_mask
    )
    # Execute query on ClickHouse (assuming `client` is already defined)
    result, columns = client.execute(query, with_column_types=True)
    all_customer_data = pd.DataFrame(result, columns=[col[0] for col in columns])
    
    # --- Step 3: Filter Data (Created Before Joining Date) ---
    if not all_customer_data.empty:
        all_customer_data['created_date'] = pd.to_datetime(all_customer_data['created_date'])
        all_customer_data['customer_request_id'] = all_customer_data['customer_request_id'].astype(int)
        final_data = all_customer_data.merge(customer_df, left_on='customer_request_id', right_on='customer_id', how='left')
        final_data = final_data[final_data['created_date'] < final_data['joining_date']]
        
        # --- Step 4: Compute Customer Metrics ---
        df = final_data.copy()
        df['date'] = df['created_date'].dt.date
        # Define delivered and request flags
        df['is_delivered'] = df['status'].isin(['DELIVERED']).astype(int)
        df['is_request'] = df['status'].isin(['DELIVERED', 'CANCELLED']).astype(int)
        
        # Aggregate basic metrics per customer_request_id
        agg_df = df.groupby('customer_request_id').agg(
            first_order_date=('created_date', 'min'),
            last_order_date=('created_date', 'max'),
            delivered_orders=('is_delivered', 'sum'),
            total_requests=('is_request', 'sum'),
            request_days=('date', 'nunique')
        ).reset_index()
        
        # Compute recency (days since last order)
        reference_date = df['created_date'].max()
        agg_df['recency'] = (reference_date - agg_df['last_order_date']).dt.days
        
        # Compute active period (period_days) and period_weeks
        agg_df['period_days'] = (agg_df['last_order_date'] - agg_df['first_order_date']).dt.days + 1
        agg_df.loc[agg_df['period_days'] <= 0, 'period_days'] = 1
        agg_df['period_weeks'] = agg_df['period_days'] / 7.0
        
        # Basic daily metrics
        agg_df['avg_delivered_daily'] = agg_df['delivered_orders'] / agg_df['period_days']
        agg_df['avg_requests_daily'] = agg_df['total_requests'] / agg_df['period_days']
        # (Other frequency metrics are not needed since our clustering features come from our new method)
        
        # Additional Behavioral Metrics:
        # 1. Variance in delivered orders per day
        daily_delivered = df.groupby(['customer_request_id', 'date']).agg(
            daily_delivered=('is_delivered', 'sum')
        ).reset_index()
        variance_df = daily_delivered.groupby('customer_request_id')['daily_delivered'].var().reset_index()
        variance_df.rename(columns={'daily_delivered': 'variance_delivered_daily'}, inplace=True)
        variance_df['variance_delivered_daily'] = variance_df['variance_delivered_daily'].fillna(0)
        agg_df = pd.merge(agg_df, variance_df, on='customer_request_id', how='left')
        
        # 2. Average time between delivered orders
        agg_df['avg_time_between_delivered'] = np.where(
            agg_df['delivered_orders'] > 1,
            agg_df['period_days'] / (agg_df['delivered_orders'] - 1),
            agg_df['period_days']
        )
        
        # --- Step 5: Assign Clusters ---
        # Use the same feature set as in the clustering model:
        feature_columns = [
            "recency",
            "period_days",
            "delivered_orders",
            "avg_delivered_daily",
            "variance_delivered_daily",
            "avg_time_between_delivered",
            "avg_requests_daily"
        ]
        
        # Read the cluster summary from Sales_Clustering.Clusters_Summary
        cluster_summary = pd.read_sql("SELECT * FROM Sales_Clustering.Clusters_Summary", engine)
        cluster_summary.columns = cluster_summary.columns.str.strip()
        agg_df.columns = agg_df.columns.str.strip()
        
        # Extract the same feature columns for comparison
        cluster_features = cluster_summary[feature_columns].copy()
        customer_features = agg_df[feature_columns].copy()
        
        # Normalize the features (MinMaxScaler)
        scaler = MinMaxScaler()
        cluster_scaled = scaler.fit_transform(cluster_features)
        customer_scaled = scaler.transform(customer_features)
        
        # Compute Euclidean distances and assign the nearest cluster
        distances = cdist(customer_scaled, cluster_scaled, metric='euclidean')
        agg_df['assigned_cluster'] = distances.argmin(axis=1)
        
        # --- Step 6: Save Processed Data ---
        # Write the final target clustering assignments to MySQL
        agg_df.to_sql("Final_Target_Data_Clusters", con=engine, if_exists="replace", index=False)
        print("AGG Target clusters data processed and saved.")


def fetch_and_aggregate_target_data():
    """Fetches target data from ClickHouse, filters, aggregates, assigns clusters, and writes to MySQL."""
    engine = get_mysql_engine("Sales")

    # **Step 1: Get Customer Data from MySQL**
    query = "SELECT DISTINCT customer_id, joining_date FROM Sales.Locked_List ORDER BY joining_date DESC"
    customer_df = pd.read_sql(query, engine)
    customer_df['joining_date'] = pd.to_datetime(customer_df['joining_date'])
    customer_ids = customer_df['customer_id'].tolist()
    customer_mask = ','.join([f"'{customer}'" for customer in customer_ids])

    # **Step 2: Fetch Data from ClickHouse**
    with open('/opt/airflow/dags/fetch_all_target_data.sql', 'r') as file:
        sql_template = file.read()

    query = sql_template.format(
        start_date="2024-03-01",
        finish_date=datetime.today().strftime('%Y-%m-%d'),
        customer_mask=customer_mask
    )

    result, columns = client.execute(query, with_column_types=True)
    all_customer_data = pd.DataFrame(result, columns=[col[0] for col in columns])

    # **Step 3: Filter Data (Created After Joining Date)**
    if not all_customer_data.empty:
        all_customer_data['created_date'] = pd.to_datetime(all_customer_data['created_date'])
        all_customer_data['customer_request_id'] = all_customer_data['customer_request_id'].astype(int)

        final_data = all_customer_data.merge(customer_df, left_on='customer_request_id', right_on='customer_id', how='left')
        final_data = final_data[final_data['created_date'] > final_data['joining_date']]

        # **Step 4: Read `Final_Target_Data_Clusters` (df_59)**
        query = "SELECT * FROM Final_Target_Data_Clusters"
        df_59 = pd.read_sql(query, engine)

        # **Step 5: Ensure required columns exist**
        if 'customer_request_id' not in final_data.columns:
            raise ValueError("The required column 'customer_request_id' is missing from final_data.")
        if 'assigned_cluster' not in df_59.columns:
            raise ValueError("The required column 'assigned_cluster' is missing from Final_Target_Data_Clusters.")

        # **Step 6: Merge to Assign Clusters**
        final_data = final_data.merge(df_59[['customer_request_id', 'assigned_cluster']], on='customer_request_id', how='left')

        # **Step 7: Compute Ride & Request Metrics**
        final_data['request'] = final_data['ride'] + (final_data['status'] == 'CANCELLED').astype(int)
        final_data['assigned_cluster'] = final_data['assigned_cluster'].fillna(0).astype(int)

        # **Step 8: Aggregate Data by `created_date`, `customer_request_id`, `assigned_cluster` (?? and keep `joining_date`)**
        aggregated_data = final_data.groupby(['created_date', 'customer_request_id', 'assigned_cluster'], as_index=False).agg(
            rides=('ride', 'sum'),
            requests=('request', 'sum'),
            joining_date=('joining_date', 'first')  # ?? Keep the earliest `joining_date`
        )
        
        aggregated_data['created_date'] = pd.to_datetime(aggregated_data['created_date']).dt.strftime('%Y-%m-%d')
        aggregated_data['joining_date'] = pd.to_datetime(aggregated_data['joining_date']).dt.strftime('%Y-%m-%d')


        # **Step 9: Save Processed Data to MySQL**
        engine = get_mysql_engine("Sales_Clustering")
        write_to_sql_in_chunks(aggregated_data, "Aggregated_Target_Data", engine)

        print("Aggregated target data processed and saved.")




def compute_avg_daily_rides_requests():
    """Computes average daily rides and requests per cluster based on unique customers per created_date & cluster."""
    engine = get_mysql_engine("Sales_Clustering")

    # **Step 1: Read Full_customers_data**
    query = "SELECT * FROM Sales_Clustering.Full_customers_data"
    full_customers_df = pd.read_sql(query, engine)

    # **Step 2: Define request as (ride + cancelled orders)**
    full_customers_df['request'] = full_customers_df['ride'] + (full_customers_df['status'] == 'CANCELLED').astype(int)

    # **Step 3: Read Clustered_Results to get cluster assignments**
    query = "SELECT customer_request_id, cluster FROM Sales_Clustering.Clustered_Results"
    clustered_results_df = pd.read_sql(query, engine)

    # **Step 4: Merge Full_customers_data with Clustered_Results**
    full_customers_df = full_customers_df.merge(
        clustered_results_df,
        on="customer_request_id",
        how="left"
    )

    # **Step 5: Fill missing clusters with 0**
    full_customers_df['cluster'] = full_customers_df['cluster'].fillna(0).astype(int)

    # **Step 6: Convert created_date to datetime format**
    full_customers_df['created_date'] = pd.to_datetime(full_customers_df['created_date'])

    # **Step 7: Group by created_date and cluster to calculate total rides, requests, and customer count**
    cluster_metrics = full_customers_df.groupby(['created_date', 'cluster']).agg(
        total_rides=('ride', 'sum'),
        total_requests=('request', 'sum'),
        unique_customers=('customer_request_id', 'nunique')  # Count unique customers per cluster
    ).reset_index()

    # **Step 8: Compute average rides and requests per cluster**
    cluster_metrics['avg_rides_per_customer'] = cluster_metrics['total_rides'] / cluster_metrics['unique_customers']
    cluster_metrics['avg_requests_per_customer'] = cluster_metrics['total_requests'] / cluster_metrics['unique_customers']

    # **Step 9: Handle potential division by zero (if no customers in a cluster)**
    cluster_metrics['avg_rides_per_customer'].fillna(0, inplace=True)
    cluster_metrics['avg_requests_per_customer'].fillna(0, inplace=True)
    
    cluster_metrics['created_date'] = pd.to_datetime(cluster_metrics['created_date']).dt.strftime('%Y-%m-%d')
    
    # **Step 10: Save processed data to MySQL**
    write_to_sql_in_chunks(cluster_metrics, "Cluster_Avg_Daily_Metrics", engine)

    print("Cluster-level average daily rides and requests computed and saved successfully.")


def merge_lookup_view():
    # Get MySQL engine
    engine = get_mysql_engine(DB_NAME)

    # Read tables from MySQL
    cluster_avg = pd.read_sql("SELECT * FROM Sales_Clustering.Cluster_Avg_Daily_Metrics", engine)
    aggregated = pd.read_sql("SELECT * FROM Sales_Clustering.Aggregated_Target_Data", engine)

    # Ensure created_date and joining_date are in a consistent format (date-only)
    cluster_avg['created_date'] = pd.to_datetime(cluster_avg['created_date']).dt.date
    aggregated['created_date'] = pd.to_datetime(aggregated['created_date']).dt.date
    aggregated['joining_date'] = pd.to_datetime(aggregated['joining_date']).dt.date

    # Convert cluster columns to integers to ensure correct merging
    aggregated['assigned_cluster'] = aggregated['assigned_cluster'].astype(int)
    cluster_avg['cluster'] = cluster_avg['cluster'].astype(int)

    # **Filter Aggregated Data to Keep Only Dates Present in Cluster_Avg_Daily_Metrics**
    valid_dates = cluster_avg['created_date'].unique()
    aggregated = aggregated[aggregated['created_date'].isin(valid_dates)]

    # Merge the tables on created_date and cluster
    merged = pd.merge(
        aggregated,
        cluster_avg[['created_date', 'cluster', 'avg_rides_per_customer', 'avg_requests_per_customer']],
        left_on=["created_date", "assigned_cluster"],
        right_on=["created_date", "cluster"],
        how="left"
    )

    # Drop the duplicate 'cluster' column after merging
    merged.drop(columns=["cluster"], inplace=True)

    # **Apply rounding to avg_rides_per_customer and avg_requests_per_customer**
    merged['avg_rides_per_customer'] = merged['avg_rides_per_customer'].round().astype('Int64')
    merged['avg_requests_per_customer'] = merged['avg_requests_per_customer'].round().astype('Int64')

    # **Calculate Incremental and Organic Rides & Requests**
    # If created_date < joining_date, all rides and requests are organic
    condition = merged['created_date'] < merged['joining_date']

    merged['incremental_rides'] = np.where(
        condition, 0, (merged['rides'] - merged['avg_rides_per_customer']).clip(lower=0)
    )
    merged['organic_rides'] = merged['rides'] - merged['incremental_rides']

    merged['incremental_requests'] = np.where(
        condition, 0, (merged['requests'] - merged['avg_requests_per_customer']).clip(lower=0)
    )
    merged['organic_requests'] = merged['requests'] - merged['incremental_requests']

    # Write the final merged result to MySQL
    merged.to_sql("Final_Lookup_View", con=engine, if_exists="replace", index=False)

    print("Final lookup view saved to Sales_Clustering.Final_Lookup_View.")


# Define tasks
fetch_all_customer_data_task = PythonOperator(
    task_id='fetch_all_customer_data',
    python_callable=fetch_all_customer_data,
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag
)

calculate_metrics_and_cluster_task = PythonOperator(
    task_id='calculate_metrics_and_cluster',
    python_callable=calculate_metrics_and_cluster,
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag
)

fetch_all_target_data_task = PythonOperator(
    task_id='fetch_all_target_data',
    python_callable=fetch_all_target_data,
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag
)

fetch_and_aggregate_target_data_task = PythonOperator(
    task_id='fetch_and_aggregate_target_data',
    python_callable=fetch_and_aggregate_target_data,
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag
)

compute_avg_daily_rides_requests_task = PythonOperator(
    task_id='compute_avg_daily_rides_requests',
    python_callable=compute_avg_daily_rides_requests,
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag
)


merge_lookup_view_task = PythonOperator(
    task_id='merge_lookup_view',
    python_callable=merge_lookup_view,
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag
)


# **Set Task Dependencies**
fetch_all_customer_data_task >> calculate_metrics_and_cluster_task >> fetch_all_target_data_task >> fetch_and_aggregate_target_data_task >> compute_avg_daily_rides_requests_task >> merge_lookup_view_task


