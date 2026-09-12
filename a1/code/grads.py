import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 1
    λ = 4  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i ** λ
    return result

def foo_grad(x):
    return 4 * (x ** 3)

def bar(x):
    return np.prod(x)

def bar_grad(x):
    #To do the gradient, just mult all but current in the vector
    numZeros = np.sum(x == 0) #lets us use the fast way
    if numZeros == 0:
        return np.prod(x) / x
    elif numZeros == 1: 
        indZero = np.argmax(x == 0) #finding the zero
        grad = np.zeros_like(x, dtype=float) #Vector of zeros b/c product would be 0 for the two that aren't zeros
        grad[indZero] = np.prod(np.delete(x, indZero)) #Asked claude sonnet 5 "how can I handle cases where there is one zero"
        return grad
    else:
        return np.zeros_like(x, dtype=float) #if more than one zero 
                                             #whole vector becomes zero
    # Your implementation here...
    # Hint: This is a bit tricky - what if one of the x[i] is zero?
    #print("TODO: Not implemented yet")


