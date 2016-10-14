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

class RecursiveLinearFilterNode1d(VisionNode):
    """
    This node creates an exponential filter on a series of images.

    It operates on the first dimension of 3d tensors and the second dimension of 5d tensors.
    """
    def __init__(self,retina=None,config=None,name=None,input_variable=None): 
        self.retina = retina 
        self.config = config
        self.state = None
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        # 3d version
        self._I = T.dtensor3(self.name+"_I")
        self._preceding_V = T.dmatrix(self.name+"_preceding_V") # initial condition for sequence
        self._b_0 = T.dscalar(self.name+"_b_0")
        self._a_0 = T.dscalar(self.name+"_a_0")
        self._a_1 = T.dscalar(self.name+"_a_1")
        self._k = T.iscalar(self.name+"_k_bip") # number of iteration steps
        def bipolar_step(input_image,
                        preceding_V,b_0, a_0, a_1):
            V = (input_image * b_0 - preceding_V * a_1) / a_0
            return V

        # The order in theano.scan has to match the order of arguments in the function bipolar_step
        self._result, self._updates = theano.scan(fn=bipolar_step,
                                      outputs_info=[self._preceding_V],
                                      sequences = [self._I],
                                      non_sequences=[self._b_0, self._a_0, self._a_1],
                                      n_steps=self._k)
        self.output_varaible = self._result[0]
        # The order of arguments presented here is arbitrary (will be inferred by the symbols provided),
        #  but function calls to compute_V_bip have to match this order!
        self.compute_V = theano.function(inputs=[self._I,self._preceding_V,
                                                      self._b_0, self._a_0, self._a_1,
                                                      self._k], 
                                              outputs=self._result, 
                                              updates=self._updates)
    def create_filters(self):
        "Whenever the configuration is changed mid-run, this method has to be called again."
        tauSurround = self.retina.seconds_to_steps(self.config.get('tau__sec',0.005))
        self.controlCond_a,self.controlCond_b = retina_base.ab_filter_exp(tauSurround,self.retina.steps_to_seconds(1.0))
    def __repr__(self):
        return '[1d Rec LinFilter Node] Differential Equation'
    def run(self,input,t_start=0):
        num_I = input.reshape((input.shape[1],input.shape[3],input.shape[4]))
        if self.state is not None:
            num_preceding_V = self.state
        else:
            num_preceding_V = np.zeros((num_I.shape[1],num_I.shape[2]))
        num_k = num_I.shape[0]

        num_b_0 = self.controlCond_b[0]
        num_a_0 = self.controlCond_a[0]
        num_a_1 = self.controlCond_a[1]
        out_var = self.compute_V(
                            # input streams
                            num_I,
                            # inital frames for output streams
                            num_preceding_V,
                            # other parameters (same order as in the definition)
                            num_b_0, num_a_0, num_a_1,
                            num_k)

        self.state = out_var[-1]
        return np.array(out_var)


class RecursiveLinearFilterNode2d(VisionNode):
    """
    This node creates a 2d (variant) gaussian filter that operates on the last two dimensions of a tensor.
    """
    def __init__(self,retina=None,config=None,name=None,input_variable=None): 
        self.retina = retina 
        self.config = config
        self.state = None
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        # 3d version
        self._L = T.dtensor3(self.name+"_I")
        
        dtensor3_broadcastable = T.TensorType('float64', (False,False,True))
        self._smoothing_a1 = dtensor3_broadcastable(self.name+'smooth_a1')
        self._smoothing_a2 = dtensor3_broadcastable(self.name+'smooth_a2')
        self._smoothing_a3 = dtensor3_broadcastable(self.name+'smooth_a3')
        self._smoothing_a4 = dtensor3_broadcastable(self.name+'smooth_a4')
        self._smoothing_b1 = dtensor3_broadcastable(self.name+'smooth_b1')
        self._smoothing_b2 = dtensor3_broadcastable(self.name+'smooth_b2')
        
        self._kx = T.iscalar(self.name+"_kx") # number of iteration steps in x direction
        self._ky = T.iscalar(self.name+"_ky") # number of iteration steps in y direction
        def smooth_function_forward(L,a1,a2,b1,b2,Y1,Y2,I2):
            Y0 = a1 * L + a2 * I2 + b1 * Y1 + b2 * Y2
            return [Y0, Y1, L]
        def smooth_function_backward(L,Ybuf,a3,a4,b1,b2,R,Y1,Y2,I2):
            Y0 = a3 * L + a4 * I2 + b1 * Y1 + b2 * Y2
            return [Y0+Ybuf, Y0, Y1, L]
        # L has dimensions (time,x,y)
        # to iterate over x dimension we swap to (x,y,time)
        L_shuffled_to_x_y_time = self._L.dimshuffle((2,1,0))
        result_forward_x, updates = theano.scan(fn=smooth_function_forward,
                                      outputs_info = [L_shuffled_to_x_y_time[0]/2.0,
                                                      L_shuffled_to_x_y_time[0]/2.0,
                                                      L_shuffled_to_x_y_time[0]],
                                      sequences = [self._L.dimshuffle((2,1,0)),
                                                   self._smoothing_a1,
                                                   self._smoothing_a2,
                                                   self._smoothing_b1,
                                                   self._smoothing_b2],      
                                      n_steps=self._kx)
        # we again iterate over x dimension, but in the reverse dimension
        result_backward_x, updates_backward_x = theano.scan(fn=smooth_function_backward,
                                      outputs_info = [L_shuffled_to_x_y_time[-1],
                                                      L_shuffled_to_x_y_time[-1]/2.0,
                                                      L_shuffled_to_x_y_time[-1]/2.0,
                                                      L_shuffled_to_x_y_time[-1]],
                                      sequences = [L_shuffled_to_x_y_time[::-1],
                                                   result_forward_x[0][::-1],
                                                   self._smoothing_a3[::-1],
                                                   self._smoothing_a4[::-1],
                                                   self._smoothing_b1[::-1],
                                                   self._smoothing_b2[::-1]],      
                                      n_steps=self._kx)
        # result_backward_x has dimensions (x,y,time)
        # to iterate over y dimension we swap x and y to  (y,x,time)
        result_backward_x_shuffled_to_y_x_time = result_backward_x[0].dimshuffle((1,0,2))
        result_forward_y, updates_forward_y = theano.scan(fn=smooth_function_forward,
                                      outputs_info = [result_backward_x_shuffled_to_y_x_time[0,:,:]/2.0,
                                                      result_backward_x_shuffled_to_y_x_time[0,:,:]/2.0,
                                                      result_backward_x_shuffled_to_y_x_time[0,:,:]],
                                      #sequences = [result_backward_x[0].transpose(),
                                      sequences = [result_backward_x_shuffled_to_y_x_time,
                                                   self._smoothing_a1,
                                                   self._smoothing_a2,
                                                   self._smoothing_b1,
                                                   self._smoothing_b2],      
                                      n_steps=self._ky)
        result_backward_y, updates_backward_y = theano.scan(fn=smooth_function_backward,
                                      outputs_info = [result_backward_x_shuffled_to_y_x_time[-1,:,:],
                                                      result_backward_x_shuffled_to_y_x_time[-1,:,:]/2.0,
                                                      result_backward_x_shuffled_to_y_x_time[-1,:,:]/2.0,
                                                      result_backward_x_shuffled_to_y_x_time[-1,:,:]],
                                      #sequences = [result_backward_x[0].transpose()[::-1],
                                      sequences = [result_backward_x_shuffled_to_y_x_time[::-1],
                                                   result_forward_y[0][::-1],
                                                   self._smoothing_a3[::-1],
                                                   self._smoothing_a4[::-1],
                                                   self._smoothing_b1[::-1],
                                                   self._smoothing_b2[::-1]],      
                                      n_steps=self._ky)
        # result_backward_y has dimensions (y,x,time)
        # to restore the initial dimensions we have to swap to (time,x,y)
        self.smooth_x_forward = theano.function(inputs=
                                            [self._L,self._smoothing_a1,self._smoothing_a2,self._smoothing_b1,self._smoothing_b2,self._kx], 
                                            outputs=result_forward_x[0].dimshuffle((2,0,1)), updates=updates)
        self.smooth_x = theano.function(inputs=
                                            [self._L,self._smoothing_a1,self._smoothing_a2,self._smoothing_a3,self._smoothing_a4,self._smoothing_b1,self._smoothing_b2,self._kx], 
                                            outputs=result_backward_x[0].dimshuffle((2,0,1)), updates=updates_backward_x)
        self.smooth_all = theano.function(inputs=
                                            [self._L,self._smoothing_a1,self._smoothing_a2,self._smoothing_a3,self._smoothing_a4,self._smoothing_b1,self._smoothing_b2,self._kx,self._ky], 
                                            outputs=result_backward_y[0].dimshuffle((2,1,0)), updates=updates_backward_y)
    def create_filters(self):
        "Whenever the configuration is changed mid-run, this method has to be called again."
        #tauSurround = self.retina.seconds_to_steps(self.config.get('tau__sec',0.005))
        #self.controlCond_a,self.controlCond_b = retina_base.ab_filter_exp(tauSurround,self.retina.steps_to_seconds(1.0))
        pass
    def __repr__(self):
        return '[2d Gauss Filter Node] Recursive Filtering'
    def run(self,input,t_start=0):
        border = self.config.get('border',5)
        num_I = np.pad(input.reshape((input.shape[1],input.shape[3],input.shape[4])),[(0,0),(border,border),(border,border)],mode='edge')
        density = self.config.get('scalar_density',1.0) * np.ones(num_I.shape[1:])
        density = self.config.get('density_map',density)
        self.coeff = retina_base.deriche_coefficients(density)
        a1,a2,a3,a4 = [self.coeff[c][:,:,np.newaxis] for c in ['A1','A2','A3','A4']]
        b1,b2 = [self.coeff[c][:,:,np.newaxis] for c in ['B1','B2']]
        out_var = self.smooth_all(
                            # input streams
                            num_I,
                            a1,a2,a3,a4,b1,b2,
                            num_I.shape[1],
                            num_I.shape[2])
        return out_var[:,border:-border,border:-border]
        
class RecursiveLinearFilterNode3d(VisionNode):
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