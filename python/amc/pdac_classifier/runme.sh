#!/bin/bash
PROFILE=amc-poc-2024-12-20
RECAP_PROFILE=$(yq '.metadata.name' < profile/$PROFILE.yaml)

./pdac_classify $PROFILE
./pdac_recap $RECAP_PROFILE
