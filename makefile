# install python dependencies in current environment
install:
	pip install -r requirements.txt

# format python files
format:
	black .
	isort .