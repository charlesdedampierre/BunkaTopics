SHELL := /bin/bash
.PHONY : all

#############
# DEV #
#############

# python3 -m venv bunka_env
# source bunka_env/bin/activate
# rm -r bunka_env (delete the environment)

install_packages:
	pip install -e .
	pip install -e '.['dev']'

docs_serve:
	mkdocs serve

### TO PUBLISH THIS WORKS AFTER MOVING THE POETRY FILE
# build_poetry:
# python -m build --sdist --wheel

# python setup.py sdist

pypi:
	python setup.py bdist_wheel --universal

pypi_publish:
	twine upload dist/* -u __token__ -p $(PYPY_TOKEN)
### TO PUBLISH THIS WORKS

# pypi_publish_test:
# 	twine upload --repository testpypi dist/* -u __token__ -p $(PYPY_TOKEN)


see_process:
	asitop

jupyter:
	python -m jupyterlab

clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

delete_checkpoints:
	find . -type d -name ".ipynb_checkpoints" -exec rm -r {} +

format_code: clean
	black bunkatopics
	isort bunkatopics
	flake8 bunkatopics

test:
	python tests/test_bunka.py

check:
	python -m unittest tests/test_bunka.py

test_fig:
	python tests/run_bunka.py

pre_push: format_code clean check

#############
# DEV #
#############

tree_wihtout_pycache:
	tree bunkatopics -I '__pycache__'


#############
# Streamlit  #
#############

run_streamlit:
	python -m streamlit run streamlit/app.py 

#############
# scaleway  #
#############

registry__login:
	docker login $(REGISTRY) -u nologin --password-stdin <<< $(SCW_SECRET_KEY)

registry__list:
	scw registry image list --profile $(PROFILE)

container__list:
	scw container container list --profile $(PROFILE)

container__get:
	scw container container get $(ID)
