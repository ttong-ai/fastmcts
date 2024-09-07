.PHONY: help install lint test

lint:
	@black . -l 115 -t py310

test:
	black . -l 115 -t py310 --check; \
	pytest tests -v --cov=sagie --disable-warnings --cov-report term-missing
