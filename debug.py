def print_sudoku(puzzle):
    """Print a sudoku grid in ASCII format."""
    if hasattr(puzzle, 'reshape'):
        puzzle = ''.join(str(int(c)) for c in puzzle.flatten())

    print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")
    for i, c in enumerate(puzzle):
        if i % 9 == 0:
            print("|", end=" ")
        print(c if c != '0' else '.', end=" ")
        if (i + 1) % 3 == 0 and (i + 1) % 9 != 0:
            print("|", end=" ")
        if (i + 1) % 9 == 0:
            print("|")
        if (i + 1) % 27 == 0 and i < 80:
            print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")
    print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")
