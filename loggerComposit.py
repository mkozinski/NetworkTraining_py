class LoggerComposit:
    def __init__(self, loggers):
        self.loggers=loggers

    def add(self,img,output,target,l,net=None,optim=None):
        for lgr in self.loggers:
            lgr.add(img,output,target,l,net,optim)

    def logEpoch(self,net=None,optim=None,scheduler=None):
        values = []
        for k in range(0,len(self.loggers)):
            values.append(self.loggers[k].logEpoch(net,optim,scheduler))

        return values
