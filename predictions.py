import json
from textwrap import dedent

import numpy as np

from augmentations import get_train_augmentations, get_test_augmentations
from clients import BaseClient
from extraction import extract_code
from logs import get_logger
from preprocess import data2grid, colors
from sandbox import run


def get_train_prompt(data: dict) -> str:
    prompt = ''
    for i, samples in enumerate(data['train']):
        prompt += f'\n### Example {i + 1}'
        inp = data2grid(samples['input'])
        out = data2grid(samples['output'])
        prompt += f'\n**Input** - {len(inp)} x {len(inp[0].replace(' ', ''))}\n```\n'
        prompt += '\n'.join(inp) + '\n```'
        prompt += f'\n**Output** - {len(out)} x {len(out[0].replace(' ', ''))}\n```\n'
        prompt += '\n'.join(out) + '\n```\n'
    return prompt.strip()


def get_test_prompt(data: dict) -> str:
    prompt = 'Here is the test grid:'
    assert len(data['test']) == 1, 'There should be only a single test sample'
    for i, samples in enumerate(data['test']):
        inp = data2grid(samples['input'])
        prompt += f'\n**Input** - {len(inp)} x {len(inp[0].replace(' ', ''))}\n```\n'
        prompt += '\n'.join(inp) + '\n```\n'
    return prompt.strip()


def gen_hypothesis(
    client: BaseClient,
    data: dict,
    nb_hypothesis: int = 1,
) -> list[tuple[str, str]]:
    """ Generate a hypothesis (initial pattern recognition) by analyzing the training samples """
    message = dedent('''
        Can you act as an intelligent puzzle solver that notices patterns in puzzles?
        The patterns are similar to the ones in IQ tests but are always represented as colored mosaic grids.
        Each color is represented by a letter. Here are all the possible colors:
        - # stands for black (usually the background)
        - B stands for blue
        - R stands for red
        - G stands for green
        - Y stands for yellow
        - W stands for white
        - V stands for violet
        - O stands for orange
        - T stands for teal
        - P stands for pink
        \n
        \n
        Below are the puzzles:
        <puzzles>
        >>>{prompt}
        </puzzles>
        \n
        Assume that the coordinate count starts at 0. For example, the top-left cell is (0, 0).
        Below is some more information about the puzzles:
        <additional_info>
        >>>{augmentations}
        </additional_info>
        \n
        Can you analyze all the example puzzles above step-by-step, and tell what might be the pattern in the puzzles after that?
        Please DO NOT print the puzzles. Just analyze them and tell the pattern you see.
        It can be anything related to
         colors, rows, columns, shapes, heights, widths, alternating patterns, diagonals,
         connections, rotations, symmetries, mirroring, distances, filling blank spaces, counting,
         associations, wrapping, expanding, shrinking, repeating, flipping, inverting, centering, shifting,
         intersections, overlapping, unions, tetris arrangements, filling patterns with DFS,
         extrapolation of the input figures into the output, clockwise and counter-clockwise movements,
         differences, fractals, and many other patterns which you need to identify.
    ''').strip()
    # We do this to avoid issues with indentation (prompt is multi-line => it messes up with dedent)
    prompt = get_train_prompt(data)
    augmentations = get_train_augmentations(data)
    message = message.replace('>>>{prompt}', prompt)
    message = message.replace('>>>{augmentations}', augmentations)
    logger = get_logger()
    logger.debug(f'>>> USER\n{message}\n')

    predictions = client.gen_responses([message], [], nb_responses=nb_hypothesis)
    for i, prediction in enumerate(predictions):
        logger.debug(f'>>> ASSISTANT {i + 1}\n{prediction}\n')
    return [(message, prediction) for prediction in predictions]


def gen_solutions(
    client: BaseClient,
    data: dict,
    nb_hypothesis: int = 1,
    nb_predictions_per_hypothesis: int = 1,
) -> list[tuple[list[str], list[str], np.ndarray]]:
    """ Solve the puzzle using the hypothesis generated by writing a Python program """
    hypothesis = gen_hypothesis(client=client, data=data, nb_hypothesis=nb_hypothesis)
    logger = get_logger()
    logger.info(f'Generated {len(hypothesis)} hypotheses')

    # Test case analysis + Python Program
    test = data['test'][0]
    test_input_grid = [' ' + ' '.join([colors[col] for col in row]) + ' ' for row in test['input']]
    test_input_grid_str = '[\n    "' + '",\n    "'.join(test_input_grid) + '"\n]'

    res = []
    for hyp_idx, (message, response) in enumerate(hypothesis):
        messages = [message, dedent('''
            Great, following the pattern you recognized above, now have a look at the test puzzle presented below.
            >>>{test_prompt}
            \n
            Below is some more information about the puzzle:\n
            <additional_info>
            >>>{test_augmentations}
            </additional_info>
            \n
            Please provide details about how the approach would work on the test input.
            You can tell specific numbers and coordinates for this test puzzle (resulting griz size, colors, etc.).
            \n
            After that, write a Python program that would print the final output for the test input using the pattern you proposed.
            Below is a Python code that you can use as a starting point that already contains the test input grid.

            ```python
            import scipy.ndimage
            import numpy as np

            grid = >>>{test_input_grid_str}
            grid = [list(row.replace(' ', '')) for row in grid]
            grid = np.array(grid)

            # You can use all the functions of numpy and scipy.ndimage to manipulate the grid (rotations, flips, etc.)
            # You can hardcode specific patterns or coordinates if you think they are necessary
            #  For example, you can call a fill() function 5 times with different coordinates to fill parts of the grid 
            #  instead of writing a generic algorithm to detect which coordinates should initiate the calls.
            # You can use numpy functions to repeat a certain portion of an array (copy-paste, etc.).
            # Be very careful with the dimensions of the grid and the coordinates you are using to not go out of bounds.
            # You can implement a BFS algorithm to find a shortest path between two points, etc.
            # Use any algorithm you think is necessary to solve the puzzle.
            ...

            # Print the final output
            print('\\n'.join([' '.join(row) for row in grid]))
            ```

            Note that the code has to run for the test input only. So, it's totally okay to hardcode things.
            In the program, DO NOT print anything other than the final output grid.
        ''').strip()]
        messages[-1] = messages[-1].replace('>>>{test_prompt}', get_test_prompt(data))
        messages[-1] = messages[-1].replace('>>>{test_augmentations}', get_test_augmentations(data))
        messages[-1] = messages[-1].replace('>>>{test_input_grid_str}', test_input_grid_str)
        logger.debug(f'>>> USER\n{messages[-1]}\n')

        predictions = client.gen_responses(messages, [response], nb_responses=nb_predictions_per_hypothesis)
        logger.info(f'Generated {len(predictions)} predictions for {hyp_idx + 1} hypothesis')
        for pred_idx, prediction in enumerate(predictions):
            responses = [response, prediction]
            logger.debug('-' * 50)
            logger.debug(f'>>> ASSISTANT {pred_idx + 1}\n{prediction}\n')

            code = extract_code(responses[-1])
            output = run(code)
            if isinstance(output, np.ndarray):
                logger.debug('RUN OUTPUT:\n' + '\n'.join([' '.join(row) for row in output]))
            else:
                logger.debug(f'FAILED!\n{output}\n')
            res.append((messages, responses, output))
    return res


if __name__ == '__main__':
    # Hypothesis
    from clients import AnthropicClient
    client = AnthropicClient()

    with open('arc/data/training/2dee498d.json') as f:
        data = json.load(f)
        for _ in range(5):
            hyp = gen_hypothesis(client, data)
            print('-' * 50)
            for m, r in hyp:
                print(m)
                print(r)
                print('-' * 50)

    # Solutions
    # with open('arc/data/training/2dee498d.json') as f:
    #     data = json.load(f)
    #     preds = gen_solutions(client, data, nb_hypothesis=1, nb_predictions_per_hypothesis=2)
    #     print('DONE!')
    #     for pred in preds:
    #         print('\n'.join([' '.join(row) for row in pred[2]]))
    #     print('-' * 50)
    #     print('\n'.join(row.strip() for row in data2grid(data['test'][0]['output'])))
