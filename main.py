import glob
import json
import random

from clients import AnthropicClient, OpenAIClient
from logs import reset_logger
from predictions import gen_solutions
from submission import create_submission, score


if __name__ == '__main__':
    samples = list(glob.glob(f'arc/data/evaluation/*.json'))
    random.shuffle(samples)

    client = AnthropicClient()
    sub = create_submission(
        samples[:5],
        predict=lambda data: gen_solutions(client=client, data=data, nb_hypothesis=4, nb_predictions_per_hypothesis=6),
    )

    logger = reset_logger('arc')
    logger.info('Submission created successfully!\n')
    logger.debug(json.dumps(sub, indent=4))
    logger.info(f'Top 2 Accuracy: {score(sub)}')
