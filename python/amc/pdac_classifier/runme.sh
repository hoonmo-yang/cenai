#!/bin/bash
GRID=amc-poc-2024-11-14
PATH=$PATH:.

pdac_classify -b $GRID
pdac_recap -b $GRID
