# 🛫 Example Setup: Airflow with PostgreSQL for Lead Scoring System

> ⚠️ **NOTE:** All names used here (e.g., database names, project folder paths, user details) are just examples.  
> Please **replace them with your own values** according to your system or project setup.

---

## 🧰 Installation Prerequisites
```text
- OS: WSL (Ubuntu) or native Ubuntu system
- Environment: Conda environment (example name: `lead_pipeline_env`)
- Python: Version 3.8 installed
- Project Directory: (example) `~/Projects/lead_pipeline_project/src/airflow/`
```

---

## 📦 Step 1 – Install PostgreSQL
```text
sudo apt update
sudo apt install postgresql postgresql-contrib
```

---

## 🛠️ Step 2 – Configure PostgreSQL (Example Setup)
```text
# Creates a new PostgreSQL database named 'lead_db_example'
sudo -u postgres psql -c "CREATE DATABASE lead_db_example;"

# (Optional) Enter PostgreSQL CLI for advanced operations
sudo -u postgres psql
```

---

## 🧪 Step 3 – Validate the DB in CLI (Optional)
```text
\l                             # List all available databases
\c lead_db_example            # Connect to the example database
select * from uploaded_leads limit 10;  # Example table query (if table exists)
```

---

## 🐍 Step 4 – Install Airflow in Conda
```text
conda activate lead_pipeline_env  # Replace with your env name

pip install "apache-airflow==2.10.0" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.0/constraints-3.8.txt"
```

---

## 🏗️ Step 5 – Configure Shell Environment
```text
# Open shell config based on your shell
nano ~/.bashrc        # For bash users
nano ~/.zshrc         # For zsh users

# Add these lines (with your paths/envs):
export AIRFLOW_HOME=~/Projects/lead_pipeline_project/src/airflow
conda activate lead_pipeline_env

# Save and reload config
Ctrl + O → Enter → Ctrl + X
source ~/.bashrc
```

---

## 🔍 Step 6 – Verify Configuration
```text
echo $AIRFLOW_HOME     # Should output your AIRFLOW project directory
which airflow          # Should show path inside your conda env

# (Optional) Remove default Airflow directory
rm -rf ~/airflow
```

---

## 🗃️ Step 7 – Initialize Airflow DB
```text
airflow db init
```

---

## 👤 Step 8 – Create Airflow Admin User
```text
# Replace user details with your own
airflow users create \
    --username admin \
    --firstname John \
    --lastname Doe \
    --role Admin \
    --email john.doe@example.com
```

---

## 🚀 Step 9 – Run Airflow Services
```text
# Terminal 1
airflow webserver --port 8080

# Terminal 2
airflow scheduler
```

---

## 🌐 Step 10 – Open Airflow UI
```text
Go to: http://localhost:8080

Login using:
  Username: admin
  Password: (as set during user creation)
```

---

## 📁 Example Project Structure
```text
src/airflow/
├── airflow.cfg         # Configuration file
├── airflow.db          # Metadata DB (SQLite for dev only)
├── dags/               # DAGs go here
└── logs/               # Execution logs by date and task
```

---

## ⚙️ Behavior on Startup
```text
- The AIRFLOW_HOME variable is auto-loaded via shell
- Your Conda environment (example: `lead_pipeline_env`) activates automatically
- All Airflow files are kept inside `src/airflow/` for clean organization
```

---

✅ **Use this setup for example workflows like:**
- Lead scoring
- Data ingestion
- Model retraining
- Drift detection scheduling

> Let me know if you want example DAGs, PostgreSQL table schemas, or ML model integration!
