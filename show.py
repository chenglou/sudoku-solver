from datasets import load_dataset
from debug import print_sudoku

sample = load_dataset("Ritvik19/Sudoku-Dataset", split="train")[0]

print("Puzzle:")
print_sudoku(sample['puzzle'])
print("\nSolution:")
print_sudoku(sample['solution'])
