BASE_DIR = .
CENAI_ENGINE ?= cpu

include $(BASE_DIR)/include-mks/common.mk

list::
	conda $@

install::
	$(PIP) install -U -r requirements/general.txt

ifeq ($(CENAI_ENGINE),cuda)
	$(PIP) install -U -r requirements/etc_cuda.txt
	$(PIP) install -U -r requirements/llamacpp_cuda.txt
	$(PIP) install -U -r requirements/pytorch_cuda.txt
else
ifeq ($(CENAI_ENGINE),rocm)
	$(PIP) install -U -r requirements/pytorch_rocm.txt
else 
ifeq ($(CENAI_ENGINE),cpu)
	$(PIP) install -U -r requirements/etc_cpu.txt
	$(PIP) install -U -r requirements/pytorch_cpu.txt
else
	$(error unknown value of CENAI_ENGINE: $(CENAI_ENGINE))
endif
endif
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

ifeq ($(CENAI_ENGINE),cuda)
	$(PIP) uninstall -U -r requirements/etc_cuda.txt
	$(PIP) uninstall -U -r requirements/llamacpp_cuda.txt
	$(PIP) uninstall -U -r requirements/pytorch_cuda.txt
else
ifeq ($(CENAI_ENGINE),rocm)
	$(PIP) uninstall -U -r requirements/pytorch_rocm.txt
else
ifeq ($(CENAI_ENGINE),cpu)
	$(PIP) uninstall -U -r requirements/etc_cpu.txt
	$(PIP) uninstall -U -r requirements/pytorch_cpu.txt
else
	$(error unknown value of CENAI_ENGINE: $(CENAI_ENGINE))
endif
endif
endif
	@$(PIP) freeze > freeze.tmp
	@if [ ! -f freeze.txt ] || ! $(CMP) -s freeze.tmp freeze.txt; then \
		if [ -f freeze.txt ]; then \
			$(MV) freeze.txt freeze.bak; \
		fi; \
		$(MV) freeze.tmp freeze.txt; \
	fi
	@$(RM) -f freeze.tmp
