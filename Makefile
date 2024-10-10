BASE_DIR = .
CENAI_GPU ?= nvidia

include $(BASE_DIR)/include-mks/common.mk

list::
	conda $@

install::
	$(PIP) install -U -r requirements/general.txt

ifeq ($(CENAI_GPU),nvidia)
	$(PIP) install -U -r requirements/general_cuda.txt
	$(PIP) install -U -r requirements/llamacpp_cuda.txt
	$(PIP) install -U -r requirements/pytorch_cuda.txt
else
	$(PIP) install -U -r requirements/pytorch_rocm.txt
endif
	@$(PIP) freeze > freeze.tmp
	@if [ ! -f freeze.txt ] || ! $(CMP) -s freeze.tmp freeze.txt; then \
		if [ -f freeze.txt ]; then \
			$(MV) freeze.txt freeze.bak; \
		fi; \
		$(MV) freeze.tmp freeze.txt; \
	fi
	@$(RM) -f freeze.tmp

clean:: 
	$(PIP) uninstall -U -r requirements/general.txt

ifeq ($(CENAI_GPU),nvidia)
	$(PIP) uninstall -U -r requirements/general_cuda.txt
	$(PIP) uninstall -U -r requirements/llamacpp_cuda.txt
	$(PIP) uninstall -U -r requirements/pytorch_cuda.txt
else
	$(PIP) uninstall -U -r requirements/pytorch_rocm.txt
endif
	@$(PIP) freeze > freeze.tmp
	@if [ ! -f freeze.txt ] || ! $(CMP) -s freeze.tmp freeze.txt; then \
		if [ -f freeze.txt ]; then \
			$(MV) freeze.txt freeze.bak; \
		fi; \
		$(MV) freeze.tmp freeze.txt; \
	fi
	@$(RM) -f freeze.tmp
