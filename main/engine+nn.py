import numpy as np
import math
class Value:

    def __init__(self,data,_children=(),_op='',label=''):#using children as tuple because tuple is immutable.
        self.data=data
        self.grad=0.0 
        self._prev=set(_children)
        self._op=_op
        self.label=label
        self._backward=lambda:None # this is the backward function which will be determined at runtime for each Value object. 
        #by the way we implement it, it would be only for derived nodes and not leaf nodes.

    def __repr__(self):
        return f"Value(data:{self.data})"
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other) # so that (value object) + (abstract datatype) makes sense.
        out=Value(self.data+other.data,(self,other),'+')
        def _backward(): # this fills in the derivatives of children
            self.grad+=1.0*out.grad #using += in case same node is used more than once, eg: b=a+a or (d=a+b and c=a*b) and we don't override the definition and instead
            #sum up the derivatives of all the instances of the node
            other.grad+=out.grad #adding in
        out._backward=_backward
        return out
    
    def __radd__(self,other): # fallback function for 2+(value object)
        return self+other

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out

    def __rmul__(self,other): # so that (abstract data type)*self makes sense
        return self*other
    
    def __neg__(self):
        return self*-1
    
    def __sub__(self,other):
        return self+(-other)
    
    def __rsub__(self,other):
        return self+(other)
    
    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(pow(self.data,other),(self,),f"**{other}")
        def _backward():
            self.grad+=out.grad*other*(self.data**(other-1))
        out._backward=_backward
        return out
    
    def __truediv__(self,other):
        return self*(other**-1)
    
    def __rtruediv__(self,other): #other/self
        return other * (self**-1)
    #needed to be defined even though in this case self*(other**-1) will go to rmul, but truediv will only be called in case of self/other
    # and not other/self.

    def tanh(self):
        x=self.data
        res = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out =Value(res,(self,),_op='tanh')
        def _backward():
            self.grad+=(1-(out.data**2))*out.grad
        out._backward=_backward
        return out

    def exp(self):
        out = Value(math.exp(self.data),(self,),'exp')
        def _backward():
            self.grad+=out.data*out.grad
        out._backward=_backward
        return out

    def relu(self):
        out = Value(0 if self.data<=0 else self.data,(self,),'relu')
        def _backward():
            self.grad+=out.grad*(0 if out.data==0 else 1)
        out._backward=_backward
        return out
    
    def sigmoid(self):
        res = 1/(1+math.exp(-self.data))
        out = Value(res,(self,),'sigmoid')
        def _backward():
            self.grad+=out.grad*(out.data*(1-out.data))
        out._backward=_backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

import random
class Neuron:
    def __init__(self,n_in):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1,1))
    def __call__(self,x): # call directly on the object. eg object(x)
        z = sum((wi*xi for wi,xi in zip(self.w,x)) ,self.b)
        act = z.tanh()
        return act
    def parameters(self):
        return self.w+[self.b]

class Layer:
    def __init__(self,nin,nout): # nin means input dimensions and nout means output dimensions or no of units
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self,x):
        outs=[n(x) for n in self.neurons ]
        return outs[0] if len(outs)==1 else outs
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP:
    def __init__(self,nin,nouts):
        self.layers=[]
        for out in nouts:
            self.layers.append(Layer(nin,out))
            nin=out
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0