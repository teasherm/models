.DEFAULT_GOAL := train

train:
	pythonw -m $(MODEL).main

.PHONY: train
