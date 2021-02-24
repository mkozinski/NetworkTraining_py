class LoggerGenericStdOut:
    def __init__(self, compute_logged_val):
        self.val=0
        self.count=0
        self.epoch=0
        self.min=float("inf")
        self.max=float("-inf")
        self.compute_logged_val=compute_logged_val

    def add(self,img,output,target,l,net=None,optim=None):
        self.val+=self.compute_logged_val(img,output,target,l,net,optim)
        self.count+=1
        print("computed a new val: {}, current count: {}\n"
              .format(self.val,self.count))

    def logEpoch(self,net=None,optim=None,scheduler=None):
        print("the total test val: {}\n".format(self.val/self.count))
        lastVal=self.val
        self.val=0
        self.count=0
        self.epoch+=1
        if lastVal < self.min:
            self.min=lastVal
            print("a new lowest value attained")
        if lastVal > self.max:
            self.max=lastVal
            print("a new highest value attained")
        
        return lastVal
