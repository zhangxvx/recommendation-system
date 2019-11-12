import logging.config
from os import path

log_path = path.join(path.dirname(__file__),'../log.config')
logging.config.fileConfig(log_path)
logger = logging.getLogger()
