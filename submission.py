"""
The submission module is responsible for:
 * properly processing the dataset
 * creating a properly formatted submission
 * and scoring the results

The scoring uses top-2 accuracy - like described in the ARC challenge.
For each task, we predict exactly 2 outputs for every test input grid contained in the task.
Tasks can have more than one test grid that needs a predicted output.
In those cases, we will process each grid separately.

The submission format should be as follows:
{
    "00576224": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
    "009d5c81": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
    "12997ef3": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]},
                 {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
}
The key is the task ID and the value is a list of attempts (2 attempts per task).
Each attempt is a dictionary with the key being one of `attempt1` or `attempt2` and the value being the 2D list
(representing the grid).

For a given task output, if any of the 2 predicted outputs matches the ground truth exactly (100% correct),
we score 1 for that task test output, otherwise 0.
If there are several test grids in the task, we average the score across all of them.
"""

import glob
import json
import logging
import random
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Callable, Literal

import numpy as np

from preprocess import grid2data


# Global logger
logger = logging.getLogger('arc')
logger.setLevel(logging.DEBUG)


def create_submission(
    sample_paths: list[str],
    predict: Callable[[dict], list[tuple[list[str], list[str], np.ndarray]]],
):
    print(f'There are {len(sample_paths)} samples to process...')

    submission: dict[str, list[dict[Literal['attempt_1', 'attempt_2'], list[list[int]]]]] = {}
    for path in sample_paths:
        with open(path, 'r') as f:
            task_id = path.split('/')[-1].split('.')[0]
            data = json.load(f)

            # Set up the logger specifically for this task
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()

            file_handler = logging.FileHandler(f'logs/{task_id}.txt', mode='w')
            file_handler.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - [%(levelname)s]: %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

            # Make `num_predictions` predictions for each test grid
            logger.info(f'Processing {task_id}... {len(data["test"])} test grids')
            for test in data['test']:
                prediction_data = {
                    'train': data['train'],
                    'test': [test],
                }
                predicted_grids: list[tuple[str, ...]] = []
                predictions = predict(prediction_data)
                logger.info(f'There are {len(predictions)} predictions in total')
                for messages, responses, grid in predictions:
                    logger.info(f'Messages {len(messages)}, Responses {len(responses)} => Grid {grid.shape if isinstance(grid, np.ndarray) else None}')
                    if isinstance(grid, np.ndarray):
                        predicted_grids.append(tuple([' '.join(row) for row in grid]))
                    else:
                        print('No valid grid found in the predictions. Skipping...')

                # Get the 2 most common grids (with counter) out of the predicted_grids
                most_common_grids = Counter(predicted_grids).most_common(2)
                logger.info(f'There are {len(predicted_grids)} predicted grids, out of which {len(Counter(predicted_grids))} are unique.')
                logger.debug(f'Most common grids: {most_common_grids}')

                # Convert those to data of numbers with `grid2data`
                if len(most_common_grids) > 0:
                    one = grid2data(list(most_common_grids[0][0]))
                else:
                    one = test['input']
                if len(most_common_grids) > 1:
                    two = grid2data(list(most_common_grids[1][0]))
                else:
                    two = one
                logger.debug(f'Grid 1:\n{one}')
                logger.debug(f'Grid 2:\n{two}')

                submission.setdefault(task_id, []).append({
                    'attempt_1': one,
                    'attempt_2': two,
                })

    with open('submission.json', 'w') as f:
        json.dump(submission, f, indent=2)
    logger.info('Submission created successfully!')
    return submission


def score(submission: dict[str, list[dict[Literal['attempt_1', 'attempt_2'], list[list[int]]]]]) -> float:
    """ Calculate top-2 accuracy """
    acc = 0       # Total sum of scores
    for task, predicted_tests in submission.items():
        # Read the ground truth from either `evaluation/` or `training/` folder
        if Path(f'arc/data/evaluation/{task}.json').exists():
            path = f'arc/data/evaluation/{task}.json'
        else:
            path = f'arc/data/training/{task}.json'

        with open(path, 'r') as f:
            data = json.load(f)
            task_score = 0
            for prediction_attempts, ground_truth in zip(predicted_tests, data['test']):
                ground_truth = ground_truth['output']
                if prediction_attempts['attempt_1'] == ground_truth or prediction_attempts['attempt_2'] == ground_truth:
                    task_score += 1
            acc += task_score / len(data['test'])
            print(f'Task {task} => Task score = {task_score / len(data['test'])} => acc: {acc}')
    print(f'Final score: {acc} of {len(submission)} tasks')
    return acc / len(submission)


if __name__ == '__main__':
    def solve(data: dict):
        """
        Should return:
            1. List of messages
            2. List of responses
            3. List of grids
        """
        return [
            (['Hello?'], ['Hi, how can I help?'], np.array([['R', 'G'], ['B', 'R']])),
        ]

    samples = list(glob.glob(f'arc/data/evaluation/*.json'))
    random.shuffle(samples)
    res = create_submission(samples[:5], predict=solve)
    pprint(res)

    print(score({
        '4c177718': [{
            'attempt_1': [[9]],
            'attempt_2': [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
        }, {
            'attempt_1': [[7]],
            'attempt_2': [[1]],
        }],
    }))
