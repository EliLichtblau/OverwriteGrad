import torch
from torch import Tensor
from math import prod
from typing import NamedTuple, Tuple, Callable, List, Optional

# todo fix
from torch.nn.functional import softmax

class CTXtype(NamedTuple):
    saved_tensors: Tuple[Tensor,...]
    weight: Tensor
    save_for_backward: Callable[[Tensor], None]
    summed: Optional[List]
    batch_size: int
    spec: str



class MulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CTXtype, inp: Tensor, weight: Tensor):
        if weight.requires_grad:
            ctx.save_for_backward(inp)
            ctx.weight = weight
        return inp * weight

    @staticmethod
    def backward(ctx: CTXtype, dy: Tensor):
        if not ctx.saved_tensors:
            return None, None
        inp, = ctx.saved_tensors
        weight = ctx.weight
        diff = inp.ndim - weight.ndim
        summed = list(range(diff)) + [i for i, dim in enumerate(weight.shape, diff) if dim == 1]
        weight_grad = dy * inp
        sum_grad_squared = weight_grad.square()
        if summed:
            weight_grad = weight_grad.sum(summed)
            sum_grad_squared = sum_grad_squared.sum(summed)
        weight.grad = sum_grad_squared.reshape(weight.size()) * torch.sign(weight_grad) * dy.size(0)
        #add_or_set(weight, sum_grad_squared.reshape(weight.size()) * dy.size(0))
        return dy * weight, weight_grad.reshape(weight.size())
    


class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CTXtype, inp: Tensor, weight: Tensor):
        if weight.requires_grad:
            diff = inp.ndim - weight.ndim
            ctx.summed = list(range(diff)) + [i for i, dim in enumerate(weight.shape, diff) if dim == 1]
            ctx.batch_size = inp.size(0)
            ctx.weight = weight

        return inp + weight

    @staticmethod
    def backward(ctx: CTXtype, dy: Tensor):
        if not hasattr(ctx, "weight"):
            return None, None
        weight = ctx.weight
        weight_grad = dy
        sum_grad_squared = dy.square()
        if ctx.summed:
            weight_grad = weight_grad.sum(ctx.summed)
            sum_grad_squared = sum_grad_squared.sum(ctx.summed)
        weight.grad =  sum_grad_squared.reshape(weight.size()) * torch.sign(weight_grad)* dy.size(0)
        return dy, weight_grad.reshape(weight.size())


class EinsumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: CTXtype, spec: str, inp: Tensor, weight: Tensor) -> Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(inp, weight)
            ctx.spec = spec
        return torch.einsum(spec, inp, weight).contiguous()

    @staticmethod
    def backward(ctx: CTXtype, dy: Tensor) -> Tuple[None, Tensor, Tensor]:
        if not ctx.saved_tensors:
            return None, None, None
        inp, wgt = ctx.saved_tensors
        inputs, output = ctx.spec.split('->')
        lhs, rhs = inputs.split(',')

        d_wgt = torch.einsum(f'{lhs},{output}->{rhs}', inp, dy).contiguous()
        wgt.grad = torch.einsum(f'{lhs},{output}->{rhs}', inp.square()*torch.sign(inp), dy.square()*torch.sign(dy)).contiguous()
        d_inp = torch.einsum(f"{rhs},{output}->{lhs}", wgt, dy).contiguous()

        return None, d_inp, d_wgt




# write softmax by hand




def mul(a: Tensor, b: Tensor) -> Tensor:
    return MulFn.apply(a,b)

def add(a: Tensor, b: Tensor) -> Tensor:
    return AddFn.apply(a,b)

def einsum(spec: str, a: Tensor, b: Tensor) -> Tensor:
    return EinsumFn.apply(spec, a, b)

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    batch_dims = ''.join(chr(ord('a') + i) for i in range(input.ndim - 1))
    input = einsum(f"{batch_dims}y,zy->{batch_dims}z", input, weight)
    if bias is None:
        return input
    return add(input, bias)


def matmul(inp: Tensor, wgt: Tensor):
    batch_dims = ''.join(chr(ord('a') + i) for i in range(inp.ndim - 1))
    return einsum(f"{batch_dims}y,yz->{batch_dims}z", inp, wgt)



def _convert_dims_to_positive(inp: Tensor, dims: List[int]) -> List[int]:
    """
    Dims in operations can be sent as negative - common semantic -
    converts [-1, -2, 4] -> [N_DIMS-1, N_DIMS-2, 4]
    where N_DIMS == len(inp.shape)
    """
    N_DIMS = len(inp.shape)
    dims_positive = []
    for d in dims:
        if d >= 0:
            dims_positive.append(d)
        else:
            dims_positive.append(N_DIMS-d)
    return dims_positive

def sum(inp: Tensor, dims: List[int]) -> Tensor:
    # handle full sum
    if len(dims) == 0:
        return einsum(f"{''.join(chr(ord('a')+i) for i in range(N_DIMS))}->...", inp)
    N_DIMS = len(inp.shape)
    # convert dims to positive indices
    dims_positive = _convert_dims_to_positive(inp, dims)
    
    lhs = ''.join(chr(ord('a')+i) for i in range(N_DIMS))
    rhs = [] # rhs
    for i in range(N_DIMS):
        if i not in dims_positive:
            rhs.append(chr(ord('a')+i))
    rhs = ''.join(rhs)
    return einsum(f"{lhs}->{rhs}", inp)


def mean(inp: Tensor, dims: List[int]):

    s = sum(inp,dims)
    return div(s, torch.tensor(prod(s.shape)).to(inp.device))


def neg(inp: torch.Tensor):
    neg1 = torch.tensor(-1).to(inp.device) # I think this is what we wnat
    return mul(neg1, inp)

def sub(a: Tensor, b: Tensor):
    return add(a, neg(b))

def div(a: Tensor, b: Tensor):
    # invert no grad?
    with torch.no_grad():
        _b = 1/b
    return mul(a, _b)


def var(inp: Tensor, dims: List[int]):
    # the reshape for subtraction is to replace
    # the dims the variance is computed on with 1s
    dims_positive = _convert_dims_to_positive(inp, dims)
    view = []
    for i,dim in enumerate(inp.shape):
        if i not in dims_positive:
            view.append(dim)
        else:
            view.append(i)
    # TODO: determine if view needs to be implemented as a special backprop
    # I don't think so unless we compute square grad and grad and even then
    # unsure but prolly safer to
    mu = mean(inp, dims).view(view)
    
    # compute (x - xi).square().sum()
    sub_step = sub(inp, mu)
    sigma_step = sum(mul(sub_step, sub_step), [])
    # compute div, I think n is number of entires in mu
    return div(sigma_step, torch.tensor(mu.numel()-1))


