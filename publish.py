import os
from dotenv import load_dotenv

# Load the environment file
load_dotenv()
pip_user = os.getenv("PIP_USER")
pip_password = os.getenv("PIP_PASSWORD")

os.system(f"poetry publish -u {pip_user} -p {pip_password}")
