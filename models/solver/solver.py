import torch.nn as nn
from models.physical.Fates_Functions import *

# torch.autograd.set_detect_anomaly(True)
def batchScalarJacobian_AD(x, y, graphed=True):
    # Description: extract the gradient dy/dx for scalar y (but with minibatch)
    # relying on the fact that the minibatch has nothing to do with each other!
    # y: [nb]; x: [nb, (?nx)]. For single-column x, get dydx as a vector [dy1/dx1,dy2/dx2,...,dyn/dxn]; For multiple-column x, there will be multi-column output
    # output: [nb, (nx?)]
    # if x is tuple/list, the return will also be tuple/list
    assert not (y.ndim > 1 and y.shape[-1] > 1), 'this function is only valid for batched scalar y outputs'
    gO = torch.ones_like(y, requires_grad=False)  # donno why it does not do 2D output
    dydx = torch.autograd.grad(outputs=y, inputs=x, retain_graph=True, grad_outputs=gO, create_graph=graphed)
    # calculate vjp. For the minibatch, we are taking advantage of the fact that the y at a site is unrelated to
    # the x at another site, so the matrix multiplication [1,1,...,1]*J reverts to extracting the diagonal of the Jacobian
    if isinstance(x, torch.Tensor):
        dydx = dydx[0]  # it gives a tuple
    if not graphed:
        # during test, we detach the graph
        # without doing this, the following cannot be cleaned from memory between time steps as something use them outside
        if isinstance(dydx, torch.Tensor):
            dydx = dydx.detach()
        else:
            for dd in dydx:
                dd = dd.detach()
        y = y.detach()
        gO = gO.detach()
    return dydx



class Jacobian(nn.Module):
    # DescriptionL an wrapper for all the Jacobian options -- stores some options and provide a uniform interface
    # J=Jacobian((mtd="batchScalarJacobian_AD"),(func))
    # jac=J(x,y)
    # x can be a torch.Tensor, or a tuple/list of Tensors, in which case the return will be a tuple/list of Tensors
    def __init__(self, mtd=0, func=None, create_graph=True, settings={"dx": 1e-2}):
        super(Jacobian, self).__init__()
        self.mtd = mtd
        self.func = func
        self.settings = settings
        self.create_graph = create_graph

    def forward(self, x, y):
        # adaptively select the right function
        if self.mtd == 0 or self.mtd == "batchScalarJacobian_AD":
            Jac = batchScalarJacobian_AD(x, y, graphed=self.create_graph)
        else:
            raise RuntimeError("This Jacobian method is not implemented yet")
        return Jac


def rtnobnd(x0, G, J, settings, doPrint=False):
    # Description: solves the nonlinear problem with unbounded Newton iteration
    # may have poor global convergence. but if it works for your function it should be fast.
    x = x0.clone();
    nx = 1 if x.ndim == 1 else x.shape[-1]
    iter = 0;
    ftol = 1e3;
    xtol = 1e4

    while (iter < settings["maxiter"]) and (ftol > settings["ftol"]) and (xtol > settings["xtol"]):
        f = G(x)
        if torch.isnan(f).any():
            print("True")
            break
        dfdx = J(x, f)
        if nx == 1:
            xnew = x - f / dfdx
        else:
            deltaX = torch.linalg.solve(dfdx, f)
            xnew = x - deltaX
        ftol = f.abs().max()
        xtol = (xnew - x).abs().max()
        x = xnew
        iter += 1
        if doPrint:
            print(
                f"iter={iter}, x= {float(x[0])}, dfdx= {float(dfdx[0])}, xtol= {xtol}, ftol= {ftol}")
    return x


def rtsafe(x0, G, J, lb, ub, settings, doPrint=False):
    # Description: safe newton iteration with bounds --- trial evaluations won't exceed bounds
    # mixed algorithm between newton's and midpoint
    # modified from numerical recipes http://phys.uri.edu/nigh/NumRec/bookfpdf/f9-4.pdf
    # also in PAWS. https://bitbucket.org/lbl-climate-dev/psu-paws-git/src/master/pawsPack/PAWS/src/PAWS/vdata.f90
    # solves the nonlinear problem with a range bound
    # x: [nb]

    prec = 1e-10
    x1 = torch.zeros_like(x0) + lb
    x2 = torch.zeros_like(x0) + ub
    alphaScalar = settings["alpha"];
    maxiter = settings["maxiter"];
    ftol_crit = settings["ftol"];
    xtol_crit = settings["xtol"]

    with torch.no_grad():
        # these selections do not need to be tracked, as long as the results do not
        # participate in actual computations.
        fl = G(x1)
        fh = G(x2)

    mask = fl < 0.0  # tensor can be used as numerical values
    xl = x1 * mask + (~mask) * x2
    xh = x2 * mask + (~mask) * x1

    mask = (x0 > x1) & (x0 < x2)
    x0 = (mask * x0 + (~mask) * 0.5 * (x1 + x2)).requires_grad_()
    x = x0.clone()  # avoids the leaf variable issue, and it has to be done this way -- cannot clone first and requires_grad
    f = G(x);
    dfdx = J(x, f)

    maskNull = fl * fh > 0
    dxOld = x2 - x1;
    dx = dxOld.clone();
    fOld = f.clone();

    for iter in range(maxiter):

        mask1 = f * fOld < -prec  # detect oscillations
        alpha = mask1 * alphaScalar + (~mask1) * 1.0  # attenuate the gradient to dampen oscillations
        xnew = x - alpha * f / dfdx
        mask = (((x - xh) * dfdx - f) * ((x - xl) * dfdx - f) > 0.0) | ((2.0 * f).abs() > (dxOld * dfdx).abs()) \
               | (torch.isnan(xnew)) | (dfdx.abs() < prec)
        xnewmid = 0.5 * (xl + xh)
        xnew[mask] = xnewmid[mask]  # doing ordinary mask addition does not work because NaN interacting with anything still produces NaN
        dxOld.copy_(dx)
        fOld.copy_(f)

        dx = xnew - x
        f = G(xnew); dfdx = J(xnew,f)

        ftol = f.abs().max()
        xtol = dx.abs().max()
        x.copy_(xnew)
        mask2 = f < 0
        xl = mask2 * x + (~mask2) * xl
        xh = (~mask2) * x + mask2 * xh

        iter += 1
        if doPrint:
            print(
                f"iter={iter}, x= {float(x[1])}, y= {float(f[1])}, dfdx= {float(dfdx[1])}, xtol= {xtol}, ftol= {ftol}")
        isConverged = (ftol < ftol_crit) | (xtol < xtol_crit)
        if isConverged:  # put this here rather a loop as maybe ftol will be NaN
            break

    x[maskNull] = 1e20  # root is not bounded.
    return x


class tensorNewton(nn.Module):
    # Description: solve a nonlinear problem of G(x)=0 using Newton's iteration
    # x can be a vector of unknowns. [nb, nx] where nx is the number of unknowns and nb is the minibatch size
    # minibatch is for different sites, physical parameter sets, etc.
    # model(x) should produce the residual
    def __init__(self, G, J=Jacobian(), mtd=0, lb=None, ub=None,settings={"maxiter": 10, "ftol": 1e-6, "xtol": 1e-6, "alpha": 0.75}):
        # alpha, the gradient attenuation factor, is only for some algorithms.
        super(tensorNewton, self).__init__()
        self.G = G
        self.J = J
        self.mtd = mtd
        self.lb = lb
        self.ub = ub
        self.settings = settings

    def forward(self, x0):
        if self.mtd == 0 or self.mtd == 'rtnobnd':
            return rtnobnd(x0, self.G, self.J, self.settings)
        elif self.mtd == 1 or self.mtd == 'rtsafe':
            assert self.lb is not None, 'mtd==1, using bounded rtsafe, but no upper bound is provided!'
            return rtsafe(x0, self.G, self.J, self.lb, self.ub, self.settings)
        else:
            assert self.mtd <= 1, 'tensorNewton:: the nonlinear solver has not been implemented yet'


class nonlinearsolver(torch.nn.Module):
    # Description: Includes our nonlinear system for the leaflayer photosynthesis subroutine in
    # the photosynthesis module in FATES

    # Inputs
    # PBM: the process-based model class

    def __init__(self, PBM):
        super(nonlinearsolver, self).__init__()
        self.PBM = PBM

    def forward(self, x):
        model_inputs = self.PBM.pre_fates()
        x = x.clone()
        anet, gs_mol, can_co2_ppress = self.PBM.forward_model(model_inputs, x)
        f = x - (can_co2_ppress - anet * model_inputs['can_press'] * (
                    h2o_co2_bl_diffuse_ratio * gs_mol + h2o_co2_stoma_diffuse_ratio
                    * model_inputs['gb_mol']) / (model_inputs['gb_mol'] * gs_mol))

        x_init = self.PBM.get_guess(model_inputs)
        f = torch.where(anet < 0, x - x_init, f)
        return f
















