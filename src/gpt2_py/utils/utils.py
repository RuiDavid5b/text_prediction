class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics like loss and accuracy during training and validation.
    """
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all meter attributes."""
        self.val = 0       # current value
        self.avg = 0       # running average
        self.sum = 0       # total sum
        self.count = 0     # number of observations

    def update(self, val, n=1):
        """
        Update meter with new value.

        Args:
            val (float): current value
            n (int): number of occurrences (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} (avg: {avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
