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

dtensor5 = T.TensorType('float64', (False,)*5)

def f7(seq):
    """ This function is removing duplicates from a list while keeping the order """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

_nd_conversions = {
    0: {
        0: lambda x: x,
        1: lambda x: x.dimshuffle(('x')),
        2: lambda x: x.dimshuffle(('x','x')),
        3: lambda x: x.dimshuffle(('x','x','x')),
        4: lambda x: x.dimshuffle(('x','x','x','x')),
        5: lambda x: x.dimshuffle(('x','x','x','x','x'))
    },
    1: {
        1: lambda x: x,
        2: "Can not convert time to space.",
        3: lambda x: x.dimshuffle((0,'x','x')),
        4: "Can not convert to 4d.",
        5: lambda x: x.dimshuffle(('x',0,'x','x','x'))        
    },
    2: {
        1: "Can not convert space to time.",
        2: lambda x: x,
        3: lambda x: x.dimshuffle(('x',0,1)),
        4: "Can not convert to 4d.",
        5: lambda x: x.dimshuffle(('x','x','x',0,1))        
    },
    3: {
        1: lambda x: x.dimshuffle((0)),
        2: lambda x: x.dimshuffle((1,2)),
        3: lambda x: x,
        4: "Can not convert to 4d.",
        5: lambda x: x.dimshuffle(('x',0,'x',1,2))        
    },
    4: {},
    5: {
        1: lambda x: x.dimshuffle((1)),
        2: lambda x: x.dimshuffle((3,4)),
        3: lambda x: x.dimshuffle((1,3,4)),
        4: "Can not convert to 4d.",
        5: lambda x: x
    }
}
def make_nd(inp,dim=3):
    """
    This function reshapes 1d, 2d, 3d and 5d tensor variables into each other under the following assumptions:
    
      * a 1d tensor contains only timeseries data
      * a 2d tensor contains only spatial data
      * a 3d tensor has time as the first dimension, space as second and third
      * a 5d tensor has the dimensions (0,time,0,x,y), where 0 is an empty dimension
      
    When the input tensor already has the desired number of dimensions, it is returned.
    """
    from_d = inp.ndim
    f = _nd_conversions.get(from_d,{}).get(dim,"No valid conversion found.")
    if type(f) is str:
        raise Exception(f)
    return f(inp)

class Node(retina.vision.VisionNode):
    """
    A general vision node
    ---------------------

    This node can be combined with other nodes to do visual computations.
    
    Right now this implementation does all of its merging, searching for variables
    and building functions from one root node that is defined with a number of dependencies.

    Example for a cheap center-surround filter::

            from vision import *

            g1 = Node(name='Center 2d Filter',retina=ret,func = gauss_2d_filter, config={'scalar_density':0.9})
            e1 = Node(name='Center 1d Filter',retina=ret,func = exponential_1d_filter, config={'tau__sec':0.00001}, input_dependency=[g1])
            g2 = Node(name='Surround 2d Filter',retina=ret,func = gauss_2d_filter, config={'scalar_density':0.9}, input_dependency=[e1])
            e2 = Node(name='Surround 1d Filter',retina=ret,func = exponential_1d_filter, config={'tau__sec':0.00001}, input_dependency=[g2])

            center_surround = FunctionNode(func = lambda x,y: x-y, input_dependency=[e1,e2])
            center_surround.create_function()
            the_output = center_surround.run(the_input)

    A `Node` has to keep track of its symbolic theano variables in several sets:

        * self.parameter_variables: a list of all symbols that are used as parameters (not inputs)
        * self.state_variables: a list of the states as they appear in the computational graph in the current time step
        * self.updated_state_variables: the symbols of the state variables for the next time step (the updated variables)

    To set the inital states a list of functions which each recieve a reference to the input is stored in:

        * self.inital_states:

    Also there are two special variables that simplify combining the computational graphs of two nodes:

        * self.input_variable
        * self.output_variable

    The constructor can be given a function that accepts a symbolic variable and returns
    a dictionary of attributes that are to be updated (eg. the ouput_variable).
    """

    def __init__(self,retina=None,config=None,name=None,input_dependency=[],func=None,func_5=None,**kwargs): 
        self.model = retina 
        self.retina = retina # legacy naming, will be removed at some point
        self.config = config
        if self.config is None:
            self.config = {}
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self.input_dependency = input_dependency
        self.accept_dimensions = [3,5]
        self._updates = None
        self.collected_inputs = []
        self.node_type = 'Node'
        self.node_description = lambda: '- no computation -'
        self.parameter_variables = []
        self.state_variables = []
        self.inital_states = []
        self.updated_state_variables = []
        self.state = None
        self.compute = None
        self.update_variables = theano.updates.OrderedUpdates() ## TODO!
        self.__dict__.update(kwargs)
        if len(self.input_dependency) == 0:
            self.input_variable = T.dtensor3(self.name+"_input")
        else:
            self.input_variable = self.input_dependency[-1].output_variable
        self.output_variable = self.input_variable+0.0# define some computation here
        self.output_variable.name = self.name+'_output'
        if func is not None:
            d = func(self.input_variable,model=self.model,name=self.name,config=self.config)
            self.__dict__.update(d)
    def get_collected_inputs(self):
        if len(self.input_dependency) > 0:
            l = []
            for dependency in self.input_dependency:
                l += dependency.get_collected_inputs()
            l += self.parameter_variables + self.state_variables
            return f7(l)
        return [self.input_variable] + self.parameter_variables + self.state_variables
    def get_collected_outputs(self):
        """ optionally collect inbetween nodes? """
        l = []
        for dependency in self.input_dependency:
            l += dependency.get_collected_outputs()
        l += self.updated_state_variables
        return f7(l)
    def get_collected_updates(self):
        l = []
        for dependency in self.input_dependency:
            l += dependency.get_collected_updates()
        l += self.update_variables
        return f7(l)
    def create_function(self):
        if True:# hasattr(retina, 'debug') and retina.debug:
            print 'solving for:',self.get_collected_outputs()+[self.output_variable]
            print 'from:',self.get_collected_inputs()
            print 'updates:',self.get_collected_updates()
        self.compute_input_order = self.get_collected_inputs()
        self.compute_output_order = self.get_collected_outputs()+[self.output_variable]
        self.compute_updates_order = self.get_collected_updates()
        self.compute = theano.function(inputs=self.compute_input_order, 
                                        outputs=self.compute_output_order, 
                                        updates=self.compute_updates_order)
    def __repr__(self):
        return '['+str(self.node_type)+'] ' + str(self.node_description())
    def collect_num_inputs_list(self,num_input_variable):
        collected_num_inputs = self.get_num_inputs(num_input_variable) + self.get_states(num_input_variable)
        assert(len(self.get_states(num_input_variable)) == len(self.state_variables)), "Number of states missmatch: %r != %r" % (self.get_states(num_input_variable), (self.state_variables))
        assert(len(self.get_num_inputs(num_input_variable)) == len(self.parameter_variables)), "Number of parameters missmatch: %r != %r" % (self.get_num_inputs(num_input_variable), (self.parameter_variables))
        print (len(collected_num_inputs),len(self.parameter_variables), len(self.state_variables))
        assert(len(collected_num_inputs) == len(self.parameter_variables) + len(self.state_variables))
        if len(self.input_dependency) > 0:
            l = []
            for dependency in self.input_dependency:
                l += dependency.collect_num_inputs(num_input_variable)
            l += collected_num_inputs
            return l
        return [num_input_variable]+collected_num_inputs
    def collect_num_inputs(self,num_input_variable):
        ## TODO separate state variables from inputs
        collected_num_inputs = {}
        collected_num_inputs.update(self.get_num_inputs(num_input_variable))
        collected_num_inputs.update(dict(zip(self.state_variables,self.get_states(num_input_variable))))
        if len(self.input_dependency) > 0:
            for dependency in self.input_dependency:
                # earlier nodes can overwrite?
                collected_num_inputs.update(dependency.collect_num_inputs(num_input_variable))
            #collected_num_inputs.update(collected_num_inputs)
            return collected_num_inputs
        collected_num_inputs.update({self.input_variable: num_input_variable})
        return collected_num_inputs
    def get_states(self,num_input_variable):
        """ get states only for this node """
        if self.state is not None:
            return self.state
        else:
            return [st(num_input_variable) for st in self.inital_states]
    def set_states(self,states):
        """ set states recursively for all parent nodes """
        if len(self.updated_state_variables) > 0:
            self.state = [states[s] for s in self.updated_state_variables]
        if len(self.input_dependency) > 0:
            for dependency in self.input_dependency:
                dependency.set_states(states)
    def run(self,input,t_start=0):
        if len(input.shape) == 5 and (5 not in self.accept_dimensions) and (3 in self.accept_dimensions):
            print "Casting from 5 to 3 dimensions.."
            num_I = input.reshape((input.shape[1],input.shape[3],input.shape[4]))
        elif len(input.shape) == 3 and 3 not in self.accept_dimensions and 5 in self.accept_dimensions:
            print "Casting from 3 to 5 dimensions.."
            num_I = input[np.newaxis,:,np.newaxis,:,:]
        else:
            print "No casting required"
            num_I = input
        all_inputs = self.collect_num_inputs(num_I)
        out_var = self.compute(*[all_inputs[i] for i in self.compute_input_order])
        new_states = dict(zip(self.compute_output_order,out_var))
        self.set_states(new_states)
        return np.array(out_var[-1])
    def debug_print(self):
        theano.printing.debugprint(self.output_variable) 
    def debug_graph(self,output_file = None):
        from IPython.display import SVG, display
        if self.compute is not None:
            s = SVG(theano.printing.pydotprint(self.compute, outfile=output_file, return_image= output_file is None ,var_with_name_simple=True,
                               format='svg'))
        else:
            s = SVG(theano.printing.pydotprint(self.output_variable, outfile=output_file, return_image=output_file is None,var_with_name_simple=True,
                               format='svg'))
        if output_file is None:
            display(s)
    def get_num_inputs(self,num_input_variable):
        return {}

class FunctionNode(Node):
    """
    This node adds all its dependencies and combines the inputs into one function call.
    """

    def __init__(self,retina=None,config=None,name=None,input_dependency=[], func=lambda x: x): 
        super(FunctionNode, self).__init__(retina=retina,config=config,name=name,input_dependency=input_dependency)
        self.accept_dimensions = [3]
        if len(input_dependency) == func.func_code.co_argcount:
            self.func = func
            self.output_variable = self.func(*[dep.output_variable for dep in input_dependency])
        else:
            raise Exception("Number of inputs does not match number of outputs!")


def exponential_1d_filter(symbolic_input,model=None,name=uuid.uuid4(),config={}):
    """
        Function to be used to initialize a `Node`.

        This node implements temporal exponential filtering.

        The configuration accepts one parameter:

            * 'tau__sec': the time constant (in seconds relative to the model)

    """
    tauSurround = model.seconds_to_steps(config.get('tau__sec',0.0001))
    controlCond_a,controlCond_b = retina_base.ab_filter_exp(tauSurround,model.steps_to_seconds(1.0))

    _preceding_V = T.dmatrix(name+"_preceding_V") # initial condition for sequence
    b_0 = controlCond_b[0]
    a_0 = controlCond_a[0]
    a_1 = controlCond_a[1]
    _k = T.iscalar(name+"_k") # number of iteration steps
    def bipolar_step(input_image,
                    preceding_V):
        V = (input_image * b_0 - preceding_V * a_1) / a_0
        return V
    output_variable, _updates = theano.scan(fn=bipolar_step,
                                  outputs_info=[_preceding_V],
                                  sequences = [make_nd(symbolic_input,3)],
                                  non_sequences=[],
                                  n_steps=_k)
    output_variable.name = name+'_output'
    def get_num_inputs(num_input_variable):
        num_k = num_input_variable.shape[0]
        return dict(zip([_k],[num_k]))
    return {
        'output_variable': output_variable,
        'accept_dimensions': [3],
        'update_variables': _updates,
        'parameter_variables': [_k],
        'state_variables': [_preceding_V],
        'inital_states': [lambda num_input_variable: num_input_variable[0,:,:]],
        'updated_state_variables': [output_variable[-1]],
        'node_type': '1d Rec LinFilter Node',
        'node_description': lambda: 'Differential Equation',
        'get_num_inputs': get_num_inputs
    }

def gauss_2d_filter(symbolic_input,model=None,name=uuid.uuid4(),config={}):
    """
        Function to be used to initialize a `Node`.

        This node implements gauss filtering by two step recursion [see Deriche 92].

        The configuration accepts one of two parameters:

            * 'scalar_density': a homogeneous density of filtering
            * 'density_map': an inhomogeneous density map

        To convert $\sigma$ into density there will be some convenience functions.
    """
    dtensor3_broadcastable = T.TensorType('float64', (False,False,True))
    _smoothing_a1 = dtensor3_broadcastable(name+'smooth_a1')
    _smoothing_a2 = dtensor3_broadcastable(name+'smooth_a2')
    _smoothing_a3 = dtensor3_broadcastable(name+'smooth_a3')
    _smoothing_a4 = dtensor3_broadcastable(name+'smooth_a4')
    _smoothing_b1 = dtensor3_broadcastable(name+'smooth_b1')
    _smoothing_b2 = dtensor3_broadcastable(name+'smooth_b2')
    _kx = T.iscalar(name+"_kx") # number of iteration steps in x direction
    _ky = T.iscalar(name+"_ky") # number of iteration steps in y direction
    def smooth_function_forward(L,a1,a2,b1,b2,Y1,Y2,I2):
        Y0 = a1 * L + a2 * I2 + b1 * Y1 + b2 * Y2
        return [Y0, Y1, L]
    def smooth_function_backward(L,Ybuf,a3,a4,b1,b2,R,Y1,Y2,I2):
        Y0 = a3 * L + a4 * I2 + b1 * Y1 + b2 * Y2
        return [Y0+Ybuf, Y0, Y1, L]
    # symbolic_input has dimensions (time,x,y)
    # to iterate over x dimension we swap to (x,y,time)
    L_shuffled_to_x_y_time = make_nd(symbolic_input,3).dimshuffle((1,2,0))
    result_forward_x, updates_forward_x = theano.scan(fn=smooth_function_forward,
                                  outputs_info = [L_shuffled_to_x_y_time[0]/2.0,
                                                  L_shuffled_to_x_y_time[0]/2.0,
                                                  L_shuffled_to_x_y_time[0]],
                                  sequences = [L_shuffled_to_x_y_time,
                                               _smoothing_a1,
                                               _smoothing_a2,
                                               _smoothing_b1,
                                               _smoothing_b2],      
                                  n_steps=_kx,name=name+'_forward_pass_x')
    # we again iterate over x dimension, but in the reverse dimension
    result_backward_x, updates_backward_x = theano.scan(fn=smooth_function_backward,
                                  outputs_info = [L_shuffled_to_x_y_time[-1],
                                                  L_shuffled_to_x_y_time[-1]/2.0,
                                                  L_shuffled_to_x_y_time[-1]/2.0,
                                                  L_shuffled_to_x_y_time[-1]],
                                  sequences = [L_shuffled_to_x_y_time[::-1],
                                               result_forward_x[0][::-1],
                                               _smoothing_a3[::-1],
                                               _smoothing_a4[::-1],
                                               _smoothing_b1[::-1],
                                               _smoothing_b2[::-1]],      
                                  n_steps=_kx,name=name+'_backward_pass_x')
    # result_backward_x has dimensions (x,y,time)
    # to iterate over y dimension we swap x and y to  (y,x,time)
    result_backward_x_shuffled_to_y_x_time = result_backward_x[0].dimshuffle((1,0,2))
    result_forward_y, updates_forward_y = theano.scan(fn=smooth_function_forward,
                                  outputs_info = [result_backward_x_shuffled_to_y_x_time[0,:,:]/2.0,
                                                  result_backward_x_shuffled_to_y_x_time[0,:,:]/2.0,
                                                  result_backward_x_shuffled_to_y_x_time[0,:,:]],
                                  sequences = [result_backward_x_shuffled_to_y_x_time,
                                               _smoothing_a1.dimshuffle((1,0,2)),
                                               _smoothing_a2.dimshuffle((1,0,2)),
                                               _smoothing_b1.dimshuffle((1,0,2)),
                                               _smoothing_b2.dimshuffle((1,0,2))],      
                                  n_steps=_ky,name=name+'_forward_pass_y')
    result_backward_y, updates_backward_y = theano.scan(fn=smooth_function_backward,
                                  outputs_info = [result_backward_x_shuffled_to_y_x_time[-1,:,:],
                                                  result_backward_x_shuffled_to_y_x_time[-1,:,:]/2.0,
                                                  result_backward_x_shuffled_to_y_x_time[-1,:,:]/2.0,
                                                  result_backward_x_shuffled_to_y_x_time[-1,:,:]],
                                  sequences = [result_backward_x_shuffled_to_y_x_time[::-1],
                                               result_forward_y[0][::-1],
                                               _smoothing_a3.dimshuffle((1,0,2))[::-1],
                                               _smoothing_a4.dimshuffle((1,0,2))[::-1],
                                               _smoothing_b1.dimshuffle((1,0,2))[::-1],
                                               _smoothing_b2.dimshuffle((1,0,2))[::-1]],
                                  n_steps=_ky,name=name+'_backward_pass_y')
    update_variables = updates_forward_x + updates_backward_x + updates_forward_y + updates_backward_y
    output_variable = (result_backward_y[0].dimshuffle((2,1,0))[:,::-1,::-1]).reshape((result_backward_y[0].shape[2],result_backward_y[0].shape[1],result_backward_y[0].shape[0]))
    output_variable.name = name+'_output'
    parameter_variables = [_smoothing_a1,_smoothing_a2,_smoothing_a3,_smoothing_a4,
                           _smoothing_b1,_smoothing_b2,_kx,_ky]
    node_type = '2d Gauss Filter Node'
    node_description = lambda: 'Recursive Filtering'
    def get_num_inputs(num_input_variable):
        density = config.get('scalar_density',1.0) * np.ones(num_input_variable.shape[1:])
        density = config.get('density_map',density)
        coeff = retina_base.deriche_coefficients(density)
        a1,a2,a3,a4 = [coeff[c][:,:,np.newaxis] for c in ['A1','A2','A3','A4']]
        b1,b2 = [coeff[c][:,:,np.newaxis] for c in ['B1','B2']]
        return dict(zip(parameter_variables,[a1,a2,a3,a4,b1,b2,num_input_variable.shape[1],num_input_variable.shape[2]]))
    return {
        'output_variable': output_variable,
        'accept_dimensions': [3],
        'update_variables': update_variables,
        'parameter_variables': parameter_variables,
        'state_variables': [],
        'inital_states': [],
        'updated_state_variables': [output_variable[-1]],
        'node_type': '2d Gauss Filter Node',
        'node_description': lambda: 'Recursive Filtering',
        'get_num_inputs': get_num_inputs
    }

def kernel_3d_center_surround_filter(symbolic_input,model=None,name=uuid.uuid4(),config={}):
    """
        Function to be used to initialize a `Node`.

        Comparable to the VirtualRetina OPL layer, this node computes a center-surround signal.
        To do this it creates a big composit kernel.

    """
    _kernel = dtensor5(name+'_kernel')
    output_variable = conv3d(symbolic_input,_kernel)
    output_variable.name = name+'_output'
    parameter_variables = [_kernel]
    node_type = '3d Kernel Filter Node'
    
    epsilon = float(config.get('epsilon',0.000000001))
    num_E_n_C = m_en_filter(int(config.get('center-n__uint',0)),float(config.get('center-tau__sec',0.0001)),
                              normalize=True,retina=model)
    num_G_C = m_g_filter(float(config.get('center-sigma__deg',0.05)),float(config.get('center-sigma__deg',0.05)),
                         retina=model,normalize=True,even=False)
    num_TwuTu_C = m_t_filter(float(config.get('undershoot',{}).get('tau__sec',0.001)),
                              float(config.get('undershoot',{}).get('relative-weight',1.0)),
                              normalize=True,retina=model,epsilon=0.0000000000001)
    num_E_S = m_e_filter(float(config.get('surround-tau__sec',0.001)),retina=model,normalize=True)
    num_G_S = m_g_filter(float(config.get('surround-sigma__deg',0.15)),float(config.get('surround-sigma__deg',0.15)),
                         retina=model,normalize=True,even=False)
    num_Reshape_C_S = fake_filter(num_G_S,num_E_S)
    num_lambda_OPL = config.get('opl-amplification',0.25) / model.config.get('input-luminosity-range',255.0)
    num_w_OPL = config.get('opl-relative-weight',0.7)
    center_filter = retina_base.conv(retina_base.conv(num_E_n_C,num_TwuTu_C),
                                    num_G_C)
    num_kernel = retina_base.minimize_filter(
                        num_lambda_OPL*(
                                retina_base.conv(center_filter,num_Reshape_C_S)
                                - num_w_OPL * retina_base.conv(retina_base.conv(center_filter,num_E_S),num_G_S)),
                        filter_epsilon = epsilon)
    node_description = lambda: 'Convolution '+str(num_kernel.shape)
    def get_num_inputs(num_input_variable):
        return dict(zip(parameter_variables,[num_kernel]))
    return {
        'output_variable': output_variable,
        'accept_dimensions': [3],
        'parameter_variables': parameter_variables,
        'state_variables': [],
        'inital_states': [],
        'updated_state_variables': [],
        'node_type': '2d Gauss Filter Node',
        'node_description': lambda: 'Recursive Filtering',
        'get_num_inputs': get_num_inputs
    }

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
        L_shuffled_to_x_y_time = self._L.dimshuffle((1,2,0))
        result_forward_x, updates = theano.scan(fn=smooth_function_forward,
                                      outputs_info = [L_shuffled_to_x_y_time[0]/2.0,
                                                      L_shuffled_to_x_y_time[0]/2.0,
                                                      L_shuffled_to_x_y_time[0]],
                                      sequences = [L_shuffled_to_x_y_time,
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
                                                   self._smoothing_a1.dimshuffle((1,0,2)),
                                                   self._smoothing_a2.dimshuffle((1,0,2)),
                                                   self._smoothing_b1.dimshuffle((1,0,2)),
                                                   self._smoothing_b2.dimshuffle((1,0,2))],      
                                      n_steps=self._ky)
        result_backward_y, updates_backward_y = theano.scan(fn=smooth_function_backward,
                                      outputs_info = [result_backward_x_shuffled_to_y_x_time[-1,:,:],
                                                      result_backward_x_shuffled_to_y_x_time[-1,:,:]/2.0,
                                                      result_backward_x_shuffled_to_y_x_time[-1,:,:]/2.0,
                                                      result_backward_x_shuffled_to_y_x_time[-1,:,:]],
                                      #sequences = [result_backward_x[0].transpose()[::-1],
                                      sequences = [result_backward_x_shuffled_to_y_x_time[::-1],
                                                   result_forward_y[0][::-1],
                                                   self._smoothing_a3.dimshuffle((1,0,2))[::-1],
                                                   self._smoothing_a4.dimshuffle((1,0,2))[::-1],
                                                   self._smoothing_b1.dimshuffle((1,0,2))[::-1],
                                                   self._smoothing_b2.dimshuffle((1,0,2))[::-1]],
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
                                            outputs=result_backward_y[0].dimshuffle((2,1,0))[:,::-1,::-1], updates=updates_backward_y)
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