import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.append("..")

## Broker settings.
broker_url = "redis://redis:6379/0"

# List of modules to import when the Celery worker starts.
imports = ("api.bunka_api.jobs",)

## Using the database to store task state and results.
result_backend = "redis://redis:6379/1"
# result_backend = "db+sqlite:///results.db"

## General task settings
# task_annotations = {"tasks.add": {"rate_limit": "1/s"}}
