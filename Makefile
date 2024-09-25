BASE_DIR = .

include $(BASE_DIR)/include-mks/common.mk

list::
	conda $@

install::
	$(MV) freeze.txt freeze.bak
	$(PIP) -U install requirements.txt
	$(PIP) freeze > freeze.txt

clean:: 
	$(MV) freeze.txt freeze.bak
	$(PIP) -U uninstall requirements.txt
	$(PIP) freeze > freeze.txt
