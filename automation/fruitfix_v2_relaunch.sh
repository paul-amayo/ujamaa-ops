#!/bin/bash
# Relaunch the fruitfix pass with the final in-tree spec (no area gates)
# once the draining old-spec pass finishes. Ledgers are cached — recompiles
# are seconds; only viable blocks train.
until grep -q "FRUITFIX-ALL-DONE" /home/paperspace/logs/chain_04_fruitfix.log 2>/dev/null; do sleep 120; done
while ps -eo cmd | grep -E "ns-train" | grep -v grep > /dev/null; do sleep 60; done
mv /home/paperspace/logs/chain_04_fruitfix.log /home/paperspace/logs/chain_04_fruitfix_oldspec.log
exec /home/paperspace/code/automation/chain_04_fruitfix.sh
