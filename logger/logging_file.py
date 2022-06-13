import logging as lg
import os
# os.chdir("E:\ml project\classification project\Activity Recognition\logger")
#creating class for logging work
class define_logger:
    def __init__(self,logger_filename):
        self.name=logger_filename
    def basic_config(self):
        return lg.basicConfig(filename=f'{self.name}.log',level=lg.DEBUG,format='%(asctime)s %(message)s')
    def info(self,text):
        return lg.info(text)
    def debug(self,text):
        return lg.debug(text)
    def error(self,text):
        return lg.error(text)
    def warning(self,text):
        return lg.warning(text)
    

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

# file_handler = logging.FileHandler('sample.log')
# file_handler.setLevel(logging.ERROR)
# file_handler.setFormatter(formatter)

# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(formatter)

# logger.addHandler(file_handler)
# logger.addHandler(stream_handler)