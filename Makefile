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

api:
	uvicorn app:app --reload

access_sentence_transformers:
	cd ~/.cache/torch/sentence_transformers

access_hf_models:
	cd ~/.cache/huggingface/hub
