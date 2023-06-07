import os
import datetime
import logging
import sys

def init_logger(log_path, name="dispnet"):
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logfile = os.path.join(log_path, "%s-%s.log" % (name, datetime.datetime.today()))
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setLevel(logging.INFO)
    root.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel(logging.INFO)
    # consoleHandler.terminator = ""
    root.addHandler(consoleHandler)
    logging.debug("Logging to %s" % logfile)
