.PHONY: fmt
fmt:
	isort . \
	&& black .

.PHONY: lint
lint:
	isort --check .
	black --check .
	flake8 .
	mypy src/transformnd tests

.PHONY: test
test:
	pytest --verbose

.PHONY: install
install:
	pip install .

.PHONY: install-dev
install-dev:
	pip install -r requirements.txt \
	&& pip install -e .

.PHONY: clean-docs
clean-docs:
	rm -rf docs

.PHONY: docs
docs: clean-docs
	mkdir docs \
	&& pdoc --html --output-dir docs transformnd

.PHONY: clean-ipynb
clean-ipynb:
	jupyter nbconvert --clear-output --inplace examples/*.ipynb

.PHONY: clean
clean: clean-docs clean-ipynb

.PHONY: run-ipynb
run-ipynb:
	jupyter nbconvert --to notebook --inplace --execute examples/*.ipynb
