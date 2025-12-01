#!/usr/bin/env python3
import subprocess
import os

os.makedirs("data", exist_ok=True)
subprocess.run(["kaggle", "datasets", "download", "-d", "radcliffe/3-million-sudoku-puzzles-with-ratings", "-p", "data"])
subprocess.run(["unzip", "-o", "data/3-million-sudoku-puzzles-with-ratings.zip", "-d", "data"])
