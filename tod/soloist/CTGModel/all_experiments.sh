#!/bin/bash


## this file will run all below scripsts.

echo ">>>running-inp-ongly"
nohup bash inp.sh > 0221_bsAndDB.log &

echo ">>>running-inp-action"
nohup bash inp_action.sh > 0221_withAction.log &

echo ">>>running-inp-action-weighted"
nohup bash inp_action_weighted.sh > 0221_withAction_and_weighted.log &

echo ">>>everything done..."

