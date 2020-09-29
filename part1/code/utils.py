import numpy

def load_single(path):
    data = numpy.loadtxt(path)
    inputs = data[:,:-1]
    outputs = data[:,-1:]
    return(inputs,outputs)

def load_multi(path):
    data = numpy.loadtxt(path)
    inputs = data[:,:9]
    outputs = data[:,9:]
    return(inputs,outputs)

def load_final(path):
    data = numpy.loadtxt(path)
    inputs = data[:,:9]
    outcome = data[:,9:]
    return(inputs,outcome)