BASE_DIR = ..

include $(BASE_DIR)/include-mks/common.mk

COMPOSE_FILES = $(wildcard docker-compose-*.yaml)
DOTENV =$(CF_DIR)/.env

up::
	@for compose in $(COMPOSE_FILES); do \
		echo $(DOCKER) compose --env-file $(DOTENV) -f $$compose up -d; \
		$(DOCKER) compose --env-file $(DOTENV) -f $$compose up -d; \
	done

down::
	@for compose in $(COMPOSE_FILES); do \
		echo $(DOCKER) compose --env-file $(DOTENV) -f $$compose down; \
		$(DOCKER) compose --env-file $(DOTENV) -f $$compose down; \
	done

ps::
	@for compose in $(COMPOSE_FILES); do \
		echo $(DOCKER) compose --env-file $(DOTENV) -f $$compose ps; \
		$(DOCKER) compose --env-file $(DOTENV) -f $$compose ps; \
	done
