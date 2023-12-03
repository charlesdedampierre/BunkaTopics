import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append("..")

## Broker settings.
broker_url = "redis://redis:6379/0"

# List of modules to import when the Celery worker starts.
imports = ("api.bunka_api.jobs",)

## Using the database to store task state and results.
result_backend = 'mongodb://mongodb:27017/'
mongodb_backend_settings = {
  'database': 'bunkaapicelery',
  'taskmeta_collection': 'bunkaapicelery_taskmeta_collection',
}
# result_backend = "db+sqlite:///results.db"

task_annotations = {"tasks.add": {"rate_limit": "1/s"}}
