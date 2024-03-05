"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a,self.scalar)

    def gradient(self, out_grad, node):
        if (len(node.inputs)>1):
            raise ValueError("More Than One Input To PowerScalar Op.")
        
        return multiply(mul_scalar(out_grad,self.scalar),node.inputs[0])


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a,b)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        b = node.inputs[1]
        grad_a = divide(out_grad,b)
        grad_b = negate(divide((multiply(out_grad,a)),power_scalar(b,2)))
        print(grad_b)
        return grad_a,grad_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        #NOTE to FUTURE SELF -> Overload the division Operator
        return array_api.divide(a,self.scalar)

    def gradient(self, out_grad:Tensor, node:Tensor):
        return divide_scalar(out_grad,self.scalar)

def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def reverse_axes(self,ndim):
        axes = self.axes
        ret_list = [x for x in range(ndim)]
        temp = ret_list[axes[0]]
        ret_list[axes[0]] = ret_list[axes[1]]
        ret_list[axes[1]] = temp
        return tuple(ret_list)
    
    def axes_tuple(self):
        if self.axes is None:
            return tuple([-2,-1])
        else:
            return self.axes

    def compute(self, a):
        #NOTE
        self.axes = self.axes_tuple()
        ndim = array_api.ndim(a)        
        reverse_tuple = self.reverse_axes(ndim)
        return array_api.transpose(a,reverse_tuple)

    def gradient(self, out_grad, node):
        return transpose(out_grad,self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        if len(node.inputs)>1:
            raise ValueError("Reshape has too many inputs.")
        old_shape = array_api.shape(node.inputs[0])
        return reshape(out_grad,old_shape)

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(input_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a,self.axes)

    def gradient(self, out_grad, node):
        #So I'm guessing we'll have to average it out?
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a,b)

    def gradient(self, out_grad, node):
        a, b = node.inputs
        # Gradient w.r.t a (da):
        # print("outgrad shape", out_grad.shape)
        # print("b.T shape", transpose(b).shape)
        # print("a.T shape", transpose(a).shape)
        grad_a = matmul(out_grad,transpose(b))
        # print("a shape", a.shape)
        # print("grad_a shape", grad_a.shape)
        grad_b = matmul(transpose(a),out_grad)
        # print("b shape", b.shape)
        # print("grad_b shape", grad_b.shape)
        if (len(grad_a.shape) > len(a.shape)):
            diff = len(grad_a.shape)-len(a.shape)
            grad_a = summation(grad_a,(tuple(x for x in range(diff))))
        if (len(grad_b.shape) > len(b.shape)):
            diff = len(grad_a.shape)-len(b.shape)
            grad_b = summation(grad_b,(tuple(x for x in range(diff))))
        return grad_a, grad_b

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return mul_scalar(out_grad,-1)

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad/a


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad*exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        out = array_api.copy(a)
        out[a<0] = 0
        return out

    def gradient(self, out_grad, node):
        #What is the gradient of a ReLU?
        # 0 or 1
        out = node.realize_cached_data().copy()
        out[out>0] = 1
        return out_grad*Tensor(out)


def relu(a):
    return ReLU()(a)
