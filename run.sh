#!/bin/bash

sudo mount tmpfs fedtemp/ -t tmpfs -o  size=8G
echo "fedtemp mounted"
python3 main.py
echo "training finished"

sudo umount fedtemp/