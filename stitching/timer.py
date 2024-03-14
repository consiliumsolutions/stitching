from datetime import datetime
import logging

class Timer(object):
    def __init__(self, name):
        self.name = name
        self.tstart = datetime.now()
        self.tstop = 0
        #print("{0} started at: {1}".format(self.name, self.tstart.strftime('%H:%M')))

    def stop(self):
        self.tstop = datetime.now()
        msg = "Timer {0} took {1} secs".format(self.name, (self.tstop - self.tstart))
        print(msg)
        logging.debug(msg)
        return self.tstop - self.tstart

    def stop_silent(self):
        self.tstop = datetime.now()
        return self.tstop - self.tstart
