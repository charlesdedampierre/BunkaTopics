#!/usr/bin/env bash

set -e
set -x
 
python3 -m celery worker -l INFO -P solo
