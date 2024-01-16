#!/usr/bin/env bash

set -e
set -x
 
python -m celery worker -l INFO -P solo
