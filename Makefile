jupyter:
	python -m jupyterlab

test:
	python tests/test_bunka.py

test_fig:
	python tests/run_bunka.py

run_streamlit:
	python -m streamlit run streamlit/app.py 