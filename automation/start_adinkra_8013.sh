#!/bin/bash
# Adinkra panel-agent server on :8013 (ujamaa/adinkra), detached from any SSH session.
cd /home/paperspace/code/ujamaa/adinkra || exit 1
export ADINKRA_LEDGER=/home/paperspace/data/sankofa_demo/ledger_demo_v1.json
export HIGH_LLM_URL=http://localhost:11434
exec python3 -m uvicorn server:app --host 127.0.0.1 --port 8013 >> /home/paperspace/logs/adinkra_8013.log 2>&1
