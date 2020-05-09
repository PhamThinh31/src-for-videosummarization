import logging
from datetime import datetime
LOG_FILENAME = datetime.now().strftime('./logfile_%H_%M_%S_%d_%m_%Y.log')


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename=LOG_FILENAME,
                    filemode='a')

x= 'dep trai'
logging.error("thinh: %s"%x)