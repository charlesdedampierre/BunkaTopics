jupyter:
	python -m jupyterlab

test:
	python tests/test_bunka.py

test_fig:
	python tests/run_bunka.py

run_streamlit:
	python -m streamlit run streamlit/app.py 

format:
	black bunkatopics
	isort bunkatopics

docker_build:
	docker build -t bunkatopics .

docker_run:
	docker run -p 8000:8000 bunkatopics





