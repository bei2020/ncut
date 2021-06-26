import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('ms')
logger.setLevel('INFO')
# logger.setLevel('DEBUG')

ZEROTH=1e-16
