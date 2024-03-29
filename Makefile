.PHONY: tests

PYPI_USER := ""
PYPI_PASS := ""

setup_conda_environment:
	conda env create --file environment.yml || conda env update --file environment.yml

tests:
	pytest

build_pypi_package: tests
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	python setup.py sdist bdist_wheel

twine_upload: build_pypi_package
	@python setup.py sdist bdist_wheel
	@twine upload \
		--repository-url https://upload.pypi.org/legacy/ \
		-u $(PYPI_USER) \
		-p $(PYPI_PASS) \
		dist/*-py3-none-any.whl