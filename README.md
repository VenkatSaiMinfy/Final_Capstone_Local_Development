## 🧰 Installation Prerequisites
```text
- OS: WSL (Ubuntu) or native Ubuntu system
- Environment: Conda environment named `lead_scoring_system`
- Language: Python 3.8 installed
- Project Directory: `src/airflow/`
```

## 📦 Step 1 – Install PostgreSQL
```text
sudo apt update
sudo apt install postgresql postgresql-contrib
```

## 🛠️ Step 2 – Configure PostgreSQL
```text
# Create the database
sudo -u postgres psql -c "CREATE DATABASE lead_scoring_db;"

# (Optional) Enter PostgreSQL CLI
sudo -u postgres psql
```

## 🧪 Step 3 – Validate DB inside CLI (Optional)
```text
\l                             # List all databases
\c lead_scoring_db            # Connect to the new DB
select * from lead_data_uploaded limit 10;  # Preview table data
```

## 🐍 Step 4 – Install Airflow in Conda
```text
conda activate lead_scoring_system

pip install "apache-airflow==2.10.0" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.0/constraints-3.8.txt"
```

## 🏗️ Step 5 – Configure Shell Environment
```text
nano ~/.bashrc        # For bash users
# OR
nano ~/.zshrc         # For zsh users

# Add to bottom of file:
export AIRFLOW_HOME=~/Final_Capstone_Project/lead_scoring_project/src/airflow
conda activate lead_scoring_system

# Save and reload
Ctrl + O → Enter → Ctrl + X
source ~/.bashrc
```

## 🔍 Step 6 – Check Configuration
```text
echo $AIRFLOW_HOME
which airflow

# (Optional) Remove default Airflow folder
rm -rf ~/airflow
```

## 🗃️ Step 7 – Initialize Airflow Metadata DB
```text
airflow db init
```

## 👤 Step 8 – Create Admin User for Airflow
```text
airflow users create \
    --username admin \
    --firstname Venkat \
    --lastname Sai \
    --role Admin \
    --email venkat@example.com
```

## 🚀 Step 9 – Start Airflow Services
```text
# Terminal 1
airflow webserver --port 8080

# Terminal 2
airflow scheduler
```

## 🌐 Step 10 – Access Web Interface
```text
Visit: http://localhost:8080
```

## 📁 Project Output Structure
```text
src/airflow/
├── airflow.cfg         # Main Airflow configuration
├── airflow.db          # Metadata SQLite database
├── dags/               # DAG scripts go here
└── logs/               # Execution logs
```

## ⚙️ Environment Behavior
```text
- AIRFLOW_HOME loads automatically on terminal open
- Conda environment `lead_scoring_system` activates (if configured)
- Project files remain organized in `src/airflow/`
```
