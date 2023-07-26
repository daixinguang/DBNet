import numpy as np
import torch
import scipy.interpolate as si
def scipy_bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """
   
    count = cv.shape[0]
    degree=torch.tensor(degree)
   
    if periodic:
        kv = np.arange(-degree,count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
        degree = np.clip(degree,1,degree)

   
    else:
        degree = torch.clip(degree,torch.tensor(1),torch.tensor(count-1))
        kv = torch.clip(torch.arange(count+degree+1)-degree,torch.tensor(0),torch.tensor(count-degree))

   
    max_param = count - (degree * (1-periodic))
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0,max_param,n))