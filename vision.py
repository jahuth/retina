"""

A general set of vision model nodes


"""
import litus
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pylab as plt
from theano.tensor.nnet.conv3d2d import conv3d
from theano.tensor.signal.conv import conv2d
import uuid 

dtensor5 = T.TensorType('float64', (False,)*5)

from retina_base import conv, minimize_filter, m_t_filter, m_e_filter, m_en_filter, m_g_filter, m_g_filter_2d, fake_filter, fake_filter_shape, find_nonconflict
import retina_base
from retina_virtualretina import valid_retina_tags, RetinaConfiguration
import retina_virtualretina

class VisionNode(object):
    """ (Abstract class. All vision nodes inherit this.) """
    def __init__(self):
        self.name=""
        pass
    def plot(self):
        import matplotlib.pylab as plt
        plt.figure()
        plt.title(self.name)


class CenterSurroundLinearFilterLayerNode(VisionNode):
    """
    Since we want to have some temporal and some spatial convolutions (some 1d, some 2d, but orthogonal to each other), we have to use 3d convolution (we don't have to, but this way we never have to worry about which is which axis). 3d convolution uses 5-tensors (see: <a href="http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d">theano.tensor.nnet.conv</a>), so we define all inputs, kernels and outputs to be 5-tensors with the unused dimensions (color channels and batch/kernel number) set to be length 1.
    """
    def __init__(self,config,kernel_center=None,kernel_surround=None,name=None):
        self.config = config
        if name is None:
            name = str(uuid.uuid4())
        self.kernel_center = kernel_center if kernel_center is not None else np.ones((1,1,1,1,1))
        self.kernel_surround = kernel_surround if kernel_surround is not None else np.ones((1,1,1,1,1))
        self.name = self.config.get('name',name)
        self._I = dtensor5(name+'_I')
        self._kernel_C = dtensor5(name+'_k_C')
        self._kernel_S = dtensor5(name+'_k_S')
        self._C = conv3d(self._I,self._kernel_C)
        self._S = conv3d(self._C,self._kernel_S)
        self._Reshape_C_S = dtensor5(name+'_Reshape_C_S')
        self._lambda_OPL = T.dscalar(name+'_lambda_OPL')
        self._w_OPL = T.dscalar(name+'_lambda_OPL')
        self._I_OPL = self._lambda_OPL * (conv3d(self._C,self._Reshape_C_S) - self._w_OPL * self._S)
        self.input_variables = [self._I]
        self.internal_variables = [self._kernel_C,self._kernel_S,self._Reshape_C_S, self._lambda_OPL,self._w_OPL]
        self.output_variable = self._I_OPL
        self.compute_function= theano.function(self.input_variables + self.internal_variables, self.output_variable)
        self.num_Reshape_C_S = fake_filter(self.kernel_center)
        self.num_lambda_OPL = self.config.get('amplification',0.25) / self.config.get('input-luminosity-range',255.0)
        self.num_w_OPL = self.config.get('relative-weight',0.7)
        self.state = None
    def create_filters(self):
        self.num_Reshape_C_S = fake_filter(self.kernel_center)
        self.num_lambda_OPL = self.config.get('amplification',0.25) / self.config.get('input-luminosity-range',255.0)
        self.num_w_OPL = self.config.get('relative-weight',0.7)
    def __repr__(self):
        return '[C/S Node] Shape: '+str(fake_filter_shape(self.kernel_center,self.kernel_surround))+'\n -> C: '+str(self.kernel_center.shape)+' S: '+str(self.kernel_surround.shape)
    def run(self,input,t_start=0):
        all_filters_shape = fake_filter_shape(self.kernel_center,self.kernel_surround)
        num_L = np.pad(input.copy(),[(0,0),(0,0),(0,0),(all_filters_shape[3]/2,all_filters_shape[3]/2),(all_filters_shape[4]/2,all_filters_shape[4]/2)],mode='edge')
        if self.state is not None:
            num_L = np.concatenate([self.state,num_L],1)
        else:
            num_L = np.concatenate([[num_L[0,0,:,:,:]]*(all_filters_shape[1]-1),num_L[0,:,:,:,:]],0)[np.newaxis,:,:,:,:]
        self.state = num_L[:,-(all_filters_shape[1]-1):,:,:,:]
        return self.compute_function(num_L,self.kernel_center,self.kernel_surround,self.num_Reshape_C_S,self.num_lambda_OPL,self.num_w_OPL)
    def plot(self):
        import matplotlib.pylab as plt
        plt.figure()
        pass


class LinearFilterNode1d(VisionNode):
    """
    Since we want to have some temporal and some spatial convolutions (some 1d, some 2d, but orthogonal to each other), we have to use 3d convolution (we don't have to, but this way we never have to worry about which is which axis). 3d convolution uses 5-tensors (see: <a href="http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d">theano.tensor.nnet.conv</a>), so we define all inputs, kernels and outputs to be 5-tensors with the unused dimensions (color channels and batch/kernel number) set to be length 1.
    """

class LinearFilterNode2d(VisionNode):
    """
    Since we want to have some temporal and some spatial convolutions (some 1d, some 2d, but orthogonal to each other), we have to use 3d convolution (we don't have to, but this way we never have to worry about which is which axis). 3d convolution uses 5-tensors (see: <a href="http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d">theano.tensor.nnet.conv</a>), so we define all inputs, kernels and outputs to be 5-tensors with the unused dimensions (color channels and batch/kernel number) set to be length 1.
    """

class LinearFilterNode3d(VisionNode):
    """
    Since we want to have some temporal and some spatial convolutions (some 1d, some 2d, but orthogonal to each other), we have to use 3d convolution (we don't have to, but this way we never have to worry about which is which axis). 3d convolution uses 5-tensors (see: <a href="http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d">theano.tensor.nnet.conv</a>), so we define all inputs, kernels and outputs to be 5-tensors with the unused dimensions (color channels and batch/kernel number) set to be length 1.
    """


class ContrastGainControlNode(VisionNode):
    """
    Controls the gain with a local estimation of contrast.
    """

class SplitterNode(VisionNode):
    """
    Splits a stream into multiple streams without modifying it.
    """

class VisualSystem(object):
    def __init__(self,config):
        self.config = config
        self.channels = []
        self.process_config()
        self.output_last_t = 0
    def process_config(self):
        self.pixel_per_degree = self.config.get('pixels-per-degree',50.0)
        self.steps_per_second = 1.0/self.config.get('temporal-step__sec',1.0/1000.0)
        self.input_luminosity_range = self.config.get('input-luminosity-range',255.0)
    def degree_to_pixel(self,degree):
        return float(degree) * self.pixel_per_degree
    def pixel_to_degree(self,pixel):
        return float(pixel) / self.pixel_per_degree
    def seconds_to_steps(self,t):
        return float(t) * self.steps_per_second
    def steps_to_seconds(self,steps):
        return float(steps) / self.steps_per_second
    def print_nodes(self):
        def handle_channels(channels,indent=''):
            for c in channels:
                for node in c:
                    if type(node) is list:
                        handle_channels(node,indent+'\t')
                    else:
                        print indent+repr(node)
        handle_channels(self.channels)
    def plot_nodes(self):
        def handle_channels(channels):
            for c in channels:
                for node in c:
                    if type(node) is list:
                        handle_channels(node)
                    else:
                        node.plot()
        handle_channels(self.channels)
    def run(self,input_images,save_output=False,print_debug=False):
        """
        the vision simulator expects a tree of components in lists of lists
        """
        import datetime
        input = input_images
        starttime = datetime.datetime.now()
        if save_output:
            self.new_output = {}
            self.new_output_names = [] # we want a list, because the dict keys are unordered
        def handle_channels(channels,input):
            _input = input.copy()
            _output = []
            for c in channels:
                last_output = None
                for node in c:
                    if type(node) is list:
                        last_output = handle_channels(self.channels,_input)
                    else:
                        if print_debug:
                            print '[',datetime.datetime.now()-starttime,']', node
                            print '>> Input is:',str(_input.shape),'mean: ',np.nanmean(_input),np.mean(_input)
                        _input = node.run(_input,t_start=self.output_last_t)
                        last_output = _input
                        if save_output or node.config.get('save_output',False):
                            self.new_output[find_nonconflict(node.name,self.new_output_names)] = _input
                _output.append(last_output)
            return _output
        outputs = handle_channels(self.channels,input_images)
        if save_output:
            if print_debug:
                print "Outputs were saved for nodes: "+(", ".join(self.new_output_names))
                print "Access or remove them with the .output[..] attribute."
            for n in self.new_output_names:
                if not n in self.output:
                    self.output[n] = self.new_output[n]
                else:
                    self.output[n] = retina_base.concatenate_time(self.output[n],self.new_output[n])
            self.output_names = self.new_output_names # the order is always from the last call to run. In any case they should be identical for each call.
            self.output_last_t += input_images.shape[1]
        return outputs