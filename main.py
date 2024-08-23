import glob
import random
from pprint import pprint

from predictions import gen_solutions
from submission import create_submission, score


if __name__ == '__main__':
    samples = list(glob.glob(f'arc/data/evaluation/*.json'))
    random.shuffle(samples)
    sub = create_submission(
        samples[:2],
        predict=lambda data: gen_solutions(data, nb_hypothesis=2, nb_predictions_per_hypothesis=2),
    )
    print('-' * 50, 'SUBMISSION:', sep='\n')
    pprint(sub)
    print(score(sub))
