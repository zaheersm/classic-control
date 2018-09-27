import os
import logging
import psutil
import sys


def ensure_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
            except FileExistsError:
                # thread-safety
                pass


def setup_logger(log_file, level=logging.INFO, stdout=False):
    logger = logging.getLogger(log_file)
    formatter = logging.Formatter('%(asctime)s : %(message)s')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if stdout is True:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.setLevel(level)
    return logger

def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

