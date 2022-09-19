from torch.nn import functional as F

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
