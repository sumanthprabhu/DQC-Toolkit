.PHONY: install quality build-release clean docs

dirs_to_check := dqc tests

docs:
	mkdocs serve
quality:
	ruff check $(dirs_to_check) --select I --fix
	ruff format --check $(dirs_to_check)

build-release:
	rm -rf dist
	rm -rf build
	python3 -m pip install --upgrade build
	python3 -m build

clean:
	rm -rf __pycache__

test: 
	pytest

install-test: clean
	python -m pip install -e ".[test]"

install-dev: clean
	python -m pip install -e ".[dev]"

install: clean
	python -m pip install -e .
