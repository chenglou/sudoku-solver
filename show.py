import pandas as pd
from debug import print_sudoku

df = pd.read_csv("data/sudoku-3m.csv", nrows=10000)
sample = df[df['difficulty'] == df['difficulty'].min()].iloc[1] # relatively easy one

print(f"id: {sample['id']}, clues: {sample['clues']}, difficulty: {sample['difficulty']}")
print("\nPuzzle:")
print_sudoku(sample['puzzle'])
print("\nSolution:")
print_sudoku(sample['solution'])
