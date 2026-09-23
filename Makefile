PYTHON ?= python
OUTPUT_DIR ?= runs/mamba_forest
TRAIN_SHARDS ?= 128
VAL_SHARDS ?= 8
TEST_SHARDS ?= 64
EPOCHS ?= 40
WEIGHTS ?= $(OUTPUT_DIR)/best.weights.h5

.PHONY: install train eval test lint clean

install:
	$(PYTHON) -m pip install -e .

train:
	$(PYTHON) -m mamba_forest.train \
		--train-shards $(TRAIN_SHARDS) \
		--val-shards $(VAL_SHARDS) \
		--epochs $(EPOCHS) \
		--output-dir $(OUTPUT_DIR)

eval:
	$(PYTHON) -m mamba_forest.evaluate \
		--weights $(WEIGHTS) \
		--test-shards $(TEST_SHARDS)

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m flake8 src tests scripts --max-line-length 88

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.pyc" -delete
