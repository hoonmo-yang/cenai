BASE_DIR = .

PYTHON_VERSION = 3.12
RECIPIENT = hmyang@itcen.com

CENAI_DIR = $(PWD)
CONDA_ENV = $(notdir $(PWD))

include $(BASE_DIR)/include-mks/common.mk

info list::
	conda $@

clean::
	@if [ "$$CONDA_DEFAULT_ENV" = $(CONDA_ENV) ]; then \
		echo "ERROR: Current conda environment is $(CONDA_ENV). \
	Run 'conda deactivate' before running 'make clean'" >&2; \
	false; \
	fi

	$(CONDA) env remove -n $(CONDA_ENV) -y
	$(CONDA) create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y
	echo "Run 'conda activate $(CONDA_ENV)' and Run 'make install'"

install:: create_default_env install_basic install_engine
	@if [ -f freeze.txt ]; then \
		$(MV) freeze.txt freeze.bak; \
	fi
	@$(PIP) freeze > freeze.txt

create_default_env::
	@echo create $(BASE_DIR)/python/cenai_core/default.env
	@echo \
	export CENAI_DIR=$(CENAI_DIR)\
	> $(BASE_DIR)/python/cenai_core/default.env

install_basic::
	@if [ "$$CONDA_DEFAULT_ENV" != $(CONDA_ENV) ]; then \
		echo "ERROR: Current conda environment is $$CONDA_DEFAULT_ENV. \
	Run 'conda activate $(CONDA_ENV)' before running 'make install'" >&2; \
	false; \
	fi
	$(PIP) install -U -r requirements/basic.txt

install_engine::
	@if command -v nvidia-smi > /dev/null 2>&1; then \
		echo "CUDA version installed"; \
		$(PIP) install -U -r requirements/pytorch_cuda.txt; \
		$(CONDA) install -c pytorch -c nvidia faiss-gpu=1.9.0 -y; \
	elif [ -d "/opt/rocm" ]; then \
		echo "ROCM version installed"; \
		$(PIP) install -U -r requirements/pytorch_rocm.txt; \
		$(PIP) install -U -r requirements/other_cpu.txt; \
	else \
		echo "CPU version installed"; \
		$(PIP) install -U -r requirements/pytorch_cpu.txt; \
		$(PIP) install -U -r requirements/other_cpu.txt; \
	fi

up down ps::
	$(MAKE) -C $(DOCKER_DIR) $@

encrypt::
	@$(GPG) --encrypt --yes --recipient $(RECIPIENT) --output $(CF_DIR)/env.gpg $(CF_DIR)/.env

decrypt::
	@$(GPG) --decrypt --yes --output $(CF_DIR)/.env $(CF_DIR)/env.gpg
