✅ Prerequisites
- WSL (Ubuntu) or native Ubuntu system
- Conda environment named `lead_scoring_system`
- Python 3.8 installed and active
- Airflow project directory: `src/airflow/`

📦 Step 1: Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

🛢️ Step 2: Setup PostgreSQL Database
# Create a new database
sudo -u postgres psql -c "CREATE DATABASE lead_scoring_db;"

# (Optional) Access PostgreSQL CLI
sudo -u postgres psql

# PostgreSQL commands (inside the CLI)
\l                             # List all databases
\c lead_scoring_db            # Connect to the new DB
select * from lead_data_uploaded limit 10;  # Preview table data

🐍 Step 3: Install Airflow 2.10.0 in Conda Environment
# Make sure your Conda environment is activated
conda activate lead_scoring_system

# Install Airflow with Python 3.8 compatible constraints
pip install "apache-airflow==2.10.0" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.0/constraints-3.8.txt"

🏠 Step 4: Set AIRFLOW_HOME in Shell Config
# Open your shell configuration file
nano ~/.bashrc        # For bash users
# OR
nano ~/.zshrc         # For zsh users

# Add the following lines at the bottom
# Airflow project config
export AIRFLOW_HOME=~/Final_Capstone_Project/lead_scoring_project/src/airflow
conda activate lead_scoring_system

# Save and exit
Ctrl + O → Enter → Ctrl + X

# Reload your shell session
source ~/.bashrc

🔍 Step 5: Verify the Setup
echo $AIRFLOW_HOME
# Expected: /home/venkat/Final_Capstone_Project/lead_scoring_project/src/airflow

which airflow
# Should point to: /home/venkat/miniconda3/envs/lead_scoring_system/bin/airflow

# (Optional) Clean up the default Airflow directory
rm -rf ~/airflow

🛠️ Step 6: Initialize Airflow Metadata DB
airflow db init

👤 Step 7: Create Airflow Admin User
airflow users create \
    --username admin \
    --firstname Venkat \
    --lastname Sai \
    --role Admin \
    --email venkat@example.com

🚀 Step 8: Start Airflow Services

# Terminal 1 – Webserver
airflow webserver --port 8080

# Terminal 2 – Scheduler
airflow scheduler

# Open your browser and visit:
http://localhost:8080

📁 Output Directory Structure (after initialization)
src/airflow/
├── airflow.cfg         # Main Airflow configuration
├── airflow.db          # SQLite DB (for metadata)
├── dags/               # Folder for DAG scripts
└── logs/               # Log files for all tasks

✅ From Now On
- AIRFLOW_HOME is auto-set whenever terminal is opened
- Conda env `lead_scoring_system` auto-activates (optional but useful)
- All Airflow files (db, logs, config) are isolated inside `src/airflow/` for cleaner project management