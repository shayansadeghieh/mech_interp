.PHONY: install
install:
	@poetry config virtualenvs.in-project true
	@poetry install
