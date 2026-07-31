#!/bin/bash
# UJAMAA test suites — system python3, no GPU, no data, <1 s total.
# Run before every commit (candidate pre-commit hook for both repos).
set -e
echo "== aru_sil_core/src =="
(cd /home/paperspace/code/aru_sil_core/src && python3 -m pytest tests -q)
echo "== high =="
(cd /home/paperspace/code/high && python3 -m pytest tests -q)
echo "ALL SUITES GREEN"
