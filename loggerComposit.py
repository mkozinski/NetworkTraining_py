class LoggerComposit:
    def __init__(self, loggers):
        self.loggers=loggers

    def add(self,l,output,target):
        for lgr in self.loggers:
            lgr.add(l,output,target)

    def logEpoch(self,net):
        lastLoss=self.loggers[0].logEpoch(net)
        for k in range(1,len(self.loggers)):
            self.loggers[k].logEpoch(net)
        return lastLoss
