#!/bin/bash
# Start the NeuroFi bot and dashboard

echo "Starting Cryptobot background worker..."
python Cryptobot.py &

echo "Starting Dash dashboard..."
python dashboard.py
