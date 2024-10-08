# Names of the code directory and the Docker image, change them to match your project
DOCKER_IMAGE_TAG := real-cloned-singer-id
DOCKER_CONTAINER_NAME := real-cloned-singer-id-container
EVALUATION_DIR := evaluation
FOUNDATION_DIR := foundation
TRAINING_DIR := training
SPLITS_DIR := splits
# DATA_DIR := data

DOCKER_PARAMS= -it --rm --ipc=host --name=$(DOCKER_CONTAINER_NAME)
# Specify GPU device(s) to use. Expand shared memory size for pre-training task.
DOCKER_PARAMS+= --gpus '"device=0"' --shm-size 32G

# Run Docker container while mounting the local directory
DOCKER_RUN_MOUNT= docker run $(DOCKER_PARAMS) \
	-v $(PWD)/$(EVALUATION_DIR)/:/workspace/$(EVALUATION_DIR)/ \
	-v $(PWD)/$(FOUNDATION_DIR)/:/workspace/$(FOUNDATION_DIR)/ \
	-v $(PWD)/$(TRAINING_DIR)/:/workspace/$(TRAINING_DIR)/ \
	-v $(PWD)/$(SPLITS_DIR)/:/workspace/$(SPLITS_DIR)/ \
$(DOCKER_IMAGE_TAG)
# -v $(PWD)/$(DATA_DIR)/:/workspace/$(DATA_DIR)/

usage:
	@echo "Available commands:\n-----------"
	@echo "	build		Build the Docker image"
	@echo "	run 		Run the Docker image in a container, after building it. Then launch an interactive bash session in the container while mounting the current directory"
	@echo "	stop		Stop the container if it is running"
	@echo "	logs		Display the logs of the container"
	@echo "	attach		Attach to running container"
	@echo "	qa-check	Check if code is nice and clean"
	@echo "	qa-clean	Lint and format code"

build:
	docker build -t $(DOCKER_IMAGE_TAG) .

run: build
	$(DOCKER_RUN_MOUNT) /bin/bash

stop:
	docker stop ${DOCKER_CONTAINER_NAME} || true && docker rm ${DOCKER_CONTAINER_NAME} || true

logs:
	docker logs -f --tail 1000 $(DOCKER_CONTAINER_NAME)

attach:
	docker attach ${DOCKER_CONTAINER_NAME}

qa-check:
	poetry run mypy $(EVALUATION_DIR) $(FOUNDATION_DIR) $(TRAINING_DIR)
	poetry run ruff check --no-fix $(EVALUATION_DIR) $(FOUNDATION_DIR) $(TRAINING_DIR)
	poetry run ruff format --check $(EVALUATION_DIR) $(FOUNDATION_DIR) $(TRAINING_DIR)

qa-clean:
	poetry run ruff check --fix $(EVALUATION_DIR) $(FOUNDATION_DIR) $(TRAINING_DIR)
	poetry run ruff format $(EVALUATION_DIR) $(FOUNDATION_DIR) $(TRAINING_DIR)