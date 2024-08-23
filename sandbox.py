"""
This module should act as a sandbox to execute Python code and return the results back.
As it's specific to the ARC AGI challenge, we'll return the result grid or the raw output (for debugging).
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

import numpy as np

from process import Process

allowed_codes = {'#', 'B', 'R', 'G', 'Y', 'W', 'V', 'O', 'T', 'P'}


def run(code: str) -> np.ndarray | str:
    """
    Run the code and return the result grid if possible.
    :param code: The code
    :return: np.ndarray if the code executed successfully. Otherwise, return the raw output
    """
    if not code:
        return 'No code was provided.'

    # Create a temporary file with the code and run it with Process from process.py
    # This will allow us to limit the memory and time of the execution
    with TemporaryDirectory() as tmp_dir:
        file = Path(tmp_dir) / 'main.py'
        file.write_text(code)
        p = Process(
            command=f'python3 {file}',
            timeout=5,
            memory_limit_mb=256,
            output_limit_mb=1,
            cwd=Path(tmp_dir),
        )
        res = p.run()
        out = res.outputs.strip().replace(str(file), 'main.py')
        err = res.errors.strip().replace(str(file), 'main.py')

    if err:
        return err

    all_outputs = out + '\n' + err
    '''
    res will be a grid of values separated by spaces (and newlines for rows).
    We need to convert this to a numpy array, if possible.
    '''

    try:
        grid = [row.split() for row in out.strip().split('\n')]
        grid = np.array(grid)

        # Make sure that there is no invalid character in the grid
        for row in grid:
            for char in row:
                if char not in allowed_codes:
                    return all_outputs

        return grid
    except:
        ...

    return all_outputs


if __name__ == '__main__':
    valid = run(dedent('''
        print('R G B')
        print('B G R')
        print('G R B')
    ''').strip())
    print('Valid:', valid)

    compile_error = run(dedent('''
        print('R G B')
         print('B G R')
        print('G R B')
    ''').strip())
    print('Compile error:', compile_error)

    invalid = run(dedent('''
        print('R G B')
        print('B G R')
        print('G R')
    ''').strip())
    print('Invalid:', invalid)

    print(run(dedent('''
    import numpy as np
    \n
    grid = [
        " # # # # # G # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # # # # # # # # # # # # # # # # # # ",
        " # # # # # # # # # # Y # # # # # # # # # # # # # # # # "
    ]
    grid = [list(row.replace(' ', '')) for row in grid]
    grid = np.array(grid)
    
    # Step 1: Identify initial colored squares
    # Green (G) in row 1, column 6
    # Yellow (Y) in row 11, column 11
    
    # Step 2: Determine spacing
    # Horizontal: 5 spaces between G and Y
    # Vertical: 10 spaces between G and Y
    
    # Step 3: Extend pattern horizontally
    def fill_row(row, start_col, color):
        for col in range(start_col, len(grid[0]), 5):
            grid[row][col] = color
    
    # Fill first row with G
    fill_row(0, 5, 'G')
    
    # Fill last row with Y
    fill_row(10, 10, 'Y')
    
    # Step 4: Repeat pattern vertically
    for row in range(0, 11, 10):
        grid[row] = grid[0]  # Copy G pattern
        if row + 10 < 11:
            grid[row + 10] = grid[10]  # Copy Y pattern
    
    # Print the final output
    print('\\n'.join([' '.join(row) for row in grid]))
    ''').strip()))
