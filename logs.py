import logging

logger = logging.getLogger('arc')
logger.setLevel(logging.DEBUG)


def get_logger():
    return logger


def reset_logger(task_id: str):
    # Set up the logger specifically for this task
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(f'logs/{task_id}.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - [%(levelname)s]: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# Set up the logger to log to `arc.log` by default
reset_logger('arc')
