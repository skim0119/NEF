#* Variables
PYTHON := python3
PYTHONPATH := `pwd`

#* Installation
.PHONY: install
install:
	pip install -r requirements.txt

#* Formatters
.PHONY: formatting
formatting:
	black --config pyproject.toml ./

#* Linting
.PHONY: test
test:
	pytest -c pyproject.toml --cov=PeriodogramAnalysis --cov-fail-under=80

.PHONY: check-codestyle
check-codestyle:
	black --diff --check --config pyproject.toml ./
	# isort --diff --check-only --settings-path pyproject.toml ./

.PHONY: mypy
mypy:
	mypy --config-file pyproject.toml PeriodogramAnalysis

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove
