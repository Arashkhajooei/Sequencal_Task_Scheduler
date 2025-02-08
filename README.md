# Incremental Rides Automation

![Pipeline Diagram](https://github.com/Arashkhajooei/Sequencal_Task_Scheduler/blob/main/Diagram.png)

I want add this pictue to the readm if my gjtub repository
## Overview
The **Incremental Rides Automation** pipeline is designed to efficiently process ride data, perform clustering-based customer segmentation, and generate aggregate insights to differentiate between organic and incremental rides. This system leverages **Apache Airflow** to structure a sequence of tasks that ensure smooth data ingestion, transformation, clustering, and aggregation.

A key challenge in this project is the **absence of a proper control group**, which required an innovative approach to distinguish between organic and incremental rides. This pipeline introduces **Sequential DAG Execution**, a structured way of dynamically processing and refining data in a chain of dependent tasks.

## Pipeline Workflow
The pipeline follows a structured, sequential process within **Airflow DAGs**:


fetch_all_customer_data → calculate_metrics_and_cluster → fetch_all_target_data → fetch_and_aggregate_target_data → compute_avg_daily_rides_requests → merge_lookup_view

Each task plays a critical role in processing and structuring the data for analysis.

## Task Breakdown
### 1. Fetch All Customer Data
- Extracts **56 days** of ride data from **ClickHouse**.
- Stores results in **MySQL** for further processing.
- Uses chunked writing for efficiency.

### 2. Calculate Metrics and Cluster Customers
- Reads raw ride data and **computes behavioral metrics** such as:
  - **Recency** (days since last order)
  - **Period Days** (active period in days)
  - **Variance** (ordering consistency)
- **Removes outliers** using **Z-score filtering**.
- Applies **KMeans clustering** to segment customers based on their behavior.

### 3. Fetch and Process Target Data
- Retrieves **target customer data** from **MySQL**.
- Filters rides **before the customer’s joining date**.
- Computes the **same behavioral metrics** as in Task 2.
- Assigns each customer to the **nearest cluster** using **Euclidean distance**.

### 4. Aggregate Target Data
- Merges new customers’ ride data with their assigned clusters.
- Aggregates ride behaviors and **stores results in MySQL**.
- Computes total **rides, requests, and joining dates per cluster**.

### 5. Compute Cluster-Level Daily Metrics
- Calculates **daily average rides and requests per customer per cluster**.
- Benchmarks future behaviors for **target customers**.
- Stores results in **Cluster_Avg_Daily_Metrics**.

### 6. Merge and Generate Final Lookup Table
- Combines **historical and new customer data**.
- Merges by **created_date** and **assigned_cluster**.
- Computes **organic vs. incremental rides**:
  - **If created_date < joining_date**, rides are **organic**.
  - Otherwise, rides are **incremental**.
- Saves results to **Final_Lookup_View**.

## Technologies Used
- **Apache Airflow** for task orchestration
- **ClickHouse** for fast data querying
- **MySQL** for data storage
- **Pandas & NumPy** for data processing
- **Scikit-learn** for clustering analysis

## Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Incremental-Rides-Automation.git
   cd Incremental-Rides-Automation
