import pandas as pd
from debug import print_sudoku

sample = pd.read_csv("data/sudoku-3m.csv", nrows=1).iloc[0]

print(f"id: {sample['id']}, clues: {sample['clues']}, difficulty: {sample['difficulty']}")
print("\nPuzzle:")
print_sudoku(sample['puzzle'])
print("\nSolution:")
print_sudoku(sample['solution'])
