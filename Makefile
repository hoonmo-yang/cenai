BASE_DIR = .

include $(BASE_DIR)/include-mks/common.mk

list::
	conda $@

install::
	$(PIP) install -U -r requirements.txt
	@$(PIP) freeze > freeze.tmp
	@$(CMP) -s freeze.txt freeze.tmp || ($(MV) freeze.txt freeze.bak && $(MV) freeze.tmp freeze.txt)
	@$(RM) -f freeze.tmp

clean:: 
	$(PIP) uninstall -U -r requirements.txt
	@$(PIP) freeze > freeze.tmp
	@$(CMP) -s freeze.txt freeze.tmp || ($(MV) freeze.txt freeze.bak && $(MV) freeze.tmp freeze.txt)
	@$(RM) -f freeze.tmp
