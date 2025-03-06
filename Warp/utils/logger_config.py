import logging

def setup_logging():
    logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] [%(thread)d] %(asctime)s-%(filename)s(line: %(lineno)d) : %(message)s',
    datefmt='%H:%M:%S'
)

setup_logging()
