"""

The formulas:

$$C(x,y,t) = G * T(wu,Tu) * E(n,t) * L (x,y,t)$$
$$S(x,y,t) = G * E * C(x,y,t)$$ 
$$I_{OLP}(x,y,t) = \lambda_{OPL}(C(x,y,t) - w_{OPL} S(x,y,t)_)$$ 
$$\\\\frac{dV_{Bip}}{dt} (x,y,t) = I_{OLP}(x,y,t) - g_{A}(x,y,t)dV_{Bip}(x,y,t)$$
$$g_{A}(x,y,t) = G * E * Q(V{Bip})(x,y,t)`with $Q(V{Bip}) = g_{A}^{0} + \lambda_{A}V^2_{Bip}$$
$$I_{Gang}(x,y,t) = G * N(eT * V_{Bip})$$

with :math:`N(V) = \\\\frac{i^0_G}{1-\lambda(V-v^0_G)/i^0_G}` (if :math:`V < v^0_G`)

with :math:`N(V) = i^0_G + \lambda(V-v^0_G)` (if  :math:`V > v^0_G`)


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

class VirtualRetinaNode(object):
    """ (Abstract class. All silver retina nodes inherit this.) """
    def __init__(self):
        self.name=""
        pass
    def plot(self):
        import matplotlib.pylab as plt
        plt.figure()
        plt.title(self.name)


class VirtualRetinaOPLLayerNode(VirtualRetinaNode):
    """
    The OPL current is a filtered version of the luminance input with spatial and temporal kernels.

    $$I_{OLP}(x,y,t) = \lambda_{OPL}(C(x,y,t) - w_{OPL} S(x,y,t)_)$$

    with:

    :math:`C(x,y,t) = G * T(wu,Tu) * E(n,t) * L (x,y,t)`


    :math:`S(x,y,t) = G * E * C(x,y,t)`

    In the case of leaky heat equation:

    :math:`C(x,y,t) = T(wu,Tu) * K(sigma_C,Tau_C) * L (x,y,t)`


    :math:`S(x,y,t) = K(sigma_S,Tau_S) * C(x,y,t)`
    p.275

    To keep all dimensions similar, a *fake kernel* has to be used on the center output that contains a single 1 but has the shape of the filters used on the surround, such that the surround can be subtracted from the center.

    The inputs of the function are: 

     * :py:obj:`L` (the luminance input), 
     * :py:obj:`E_n_C`, :py:obj:`TwuTu_C`, :py:obj:`G_C` (the center filters), 
     * :py:obj:`E_S`, :py:obj:`G_S` (the surround filters), 
     * :py:obj:`Reshape_C_S` (the fake filter), 
     * :py:obj:`lambda_OPL`, :py:obj:`w_OPL` (scaling and weight parameters)

    Since we want to have some temporal and some spatial convolutions (some 1d, some 2d, but orthogonal to each other), we have to use 3d convolution (we don't have to, but this way we never have to worry about which is which axis). 3d convolution uses 5-tensors (see: <a href="http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d">theano.tensor.nnet.conv</a>), so we define all inputs, kernels and outputs to be 5-tensors with the unused dimensions (color channels and batch/kernel number) set to be length 1.
    """
    def __init__(self,retina=None,config=None,name=None):
        self.retina = retina 
        self.config = config
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self._L = dtensor5(name+'L')
        self._E_n_C = dtensor5(name+'E_n_C')
        self._TwuTu_C = dtensor5(name+'TwuTu_C')
        self._G_C = dtensor5(name+'G_C')
        self._E_S = dtensor5(name+'E_S')
        self._G_S = dtensor5(name+'G_S')
        self._Reshape_C_S = dtensor5(name+'Reshape_C_S')
        self._lambda_OPL = T.dscalar(name+'lambda_OPL')
        self._w_OPL = T.dscalar(name+'lambda_OPL')
        self._C = conv3d(conv3d(conv3d(self._L,self._E_n_C),self._TwuTu_C),self._G_C)
        self._S = conv3d(conv3d(self._C,self._E_S),self._G_S)
        self._I_OPL = self._lambda_OPL * (conv3d(self._C,self._Reshape_C_S) - self._w_OPL * self._S)
        self.input_variables = [self._L]
        self.internal_variables = [self._E_n_C,self._TwuTu_C,self._G_C,self._E_S,self._G_S,self._Reshape_C_S, self._lambda_OPL,self._w_OPL]
        self.output_variable = self._I_OPL
        self.compute_function= theano.function(self.input_variables + self.internal_variables, self.output_variable)
        self.state = None
    def create_filters(self):
        if self.config.get('leaky-heat-equation','0') == '0':
            self.num_E_n_C = m_en_filter(int(self.config.get('center-n__uint',0)),float(self.config.get('center-tau__sec')),
                                      normalize=True,retina=self.retina)
            self.num_G_C = m_g_filter(float(self.config.get('center-sigma__deg')),float(self.config.get('center-sigma__deg')),retina=self.retina,normalize=True,even=False)
        else:
            sigma_heat = float(self.config.get('center-sigma__deg'))
            tau_leak = float(self.config.get('center-tau__sec'))
            self.num_E_n_C = m_e_filter(tau_leak,retina=self.retina,normalize=True)
            gCoupling = sigma_heat**2/(2*tau_leak)
            sigma_0 = np.sqrt(2*gCoupling*self.retina.steps_to_seconds(1.0))
            self.num_G_C = m_g_filter(float(sigma_0),float(sigma_0),retina=self.retina,normalize=True,epsilon=0.000000001,even=False)
        self.num_TwuTu_C = m_t_filter(float(self.config.get('undershoot',{}).get('tau__sec')),
                                  float(self.config.get('undershoot',{}).get('relative-weight')),
                                  normalize=True,retina=self.retina,epsilon=0.0000000000001)
        self.num_E_S = m_e_filter(float(self.config.get('surround-tau__sec')),retina=self.retina,normalize=True)
        self.num_G_S = m_g_filter(float(self.config.get('surround-sigma__deg')),float(self.config.get('surround-sigma__deg')),retina=self.retina,normalize=True,even=False)
        self.num_Reshape_C_S = fake_filter(self.num_G_S,self.num_E_S)
        self.num_lambda_OPL = self.config.get('opl-amplification',0.25) / self.retina.config.get('input-luminosity-range',255.0)
        self.num_w_OPL = self.config.get('opl-relative-weight',0.7)
    def __repr__(self):
        return '[OPL Node] Shape: '+str(fake_filter_shape(self.num_E_n_C,self.num_TwuTu_C,self.num_G_C,self.num_E_S,self.num_G_S))+'\n -> C: '+str(self.num_G_C.shape)+' S: '+str(self.num_G_S.shape)
    def run(self,input,t_start=0):
        all_filters_shape = fake_filter_shape(self.num_E_n_C,self.num_TwuTu_C,self.num_G_C,self.num_E_S,self.num_G_S)
        num_L = np.pad(input.copy(),[(0,0),(0,0),(0,0),(all_filters_shape[3]/2,all_filters_shape[3]/2),(all_filters_shape[4]/2,all_filters_shape[4]/2)],mode='edge')
        if self.state is not None:
            num_L = np.concatenate([self.state,num_L],1)
        else:
            num_L = np.concatenate([[num_L[0,0,:,:,:]]*(all_filters_shape[1]-1),num_L[0,:,:,:,:]],0)[np.newaxis,:,:,:,:]
        self.state = num_L[:,-(all_filters_shape[1]-1):,:,:,:]
        return self.compute_function(num_L,self.num_E_n_C,self.num_TwuTu_C,self.num_G_C,self.num_E_S,
            self.num_G_S,self.num_Reshape_C_S,self.num_lambda_OPL,self.num_w_OPL)
    def plot(self):
        import matplotlib.pylab as plt
        plt.figure()
        plt.suptitle(self.name)
        plt.subplot(2,2,1)
        plt.title('Center')
        plt.imshow(self.num_G_C[0,0,0,:,:],interpolation='nearest',vmin=min(np.min(self.num_G_C),0),vmax=min(np.max(self.num_G_C),1))
        plt.subplot(2,2,2)
        plt.title('Center')
        plt.plot(self.num_E_n_C[0,:,0,0,0])
        plt.plot(self.num_TwuTu_C[0,:,0,0,0])
        plt.subplot(2,2,3)
        plt.title('Surround')
        plt.imshow(self.num_G_S[0,0,0,:,:],interpolation='nearest',vmin=min(np.min(self.num_G_S),0),vmax=min(np.max(self.num_G_S),1))
        plt.subplot(2,2,4)
        plt.title('Surround')
        plt.plot(self.num_E_S[0,:,0,0,0])
        plt.tight_layout()


class VirtualRetinaOPLLayerNodeLeakyHeat(VirtualRetinaNode):
    """
    The OPL current is a filtered version of the luminance input with spatial and temporal kernels.

    $$I_{OLP}(x,y,t) = \lambda_{OPL}(C(x,y,t) - w_{OPL} S(x,y,t)_)$$

    with:

    :math:`C(x,y,t) = G * T(wu,Tu) * E(n,t) * L (x,y,t)`


    :math:`S(x,y,t) = G * E * C(x,y,t)`


    To keep all dimensions similar, a *fake kernel* has to be used on the center output that contains a single 1 but has the shape of the filters used on the surround, such that the surround can be subtracted from the center.

    The inputs of the function are: 

     * :py:obj:`L` (the luminance input), 
     * :py:obj:`E_n_C`, :py:obj:`TwuTu_C`, :py:obj:`G_C` (the center filters), 
     * :py:obj:`E_S`, :py:obj:`G_S` (the surround filters), 
     * :py:obj:`Reshape_C_S` (the fake filter), 
     * :py:obj:`lambda_OPL`, :py:obj:`w_OPL` (scaling and weight parameters)

    Since we want to have some temporal and some spatial convolutions (some 1d, some 2d, but orthogonal to each other), we have to use 3d convolution (we don't have to, but this way we never have to worry about which is which axis). 3d convolution uses 5-tensors (see: <a href="http://deeplearning.net/software/theano/library/tensor/nnet/conv.html#theano.tensor.nnet.conv3d2d.conv3d">theano.tensor.nnet.conv</a>), so we define all inputs, kernels and outputs to be 5-tensors with the unused dimensions (color channels and batch/kernel number) set to be length 1.
    """
    def __init__(self,retina=None,config=None,name=None):
        self.retina = retina 
        self.config = config
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self._L = dtensor5(name+'L')
        self._E_n_C = dtensor5(name+'E_n_C')
        self._TwuTu_C = dtensor5(name+'TwuTu_C')
        self._G_C = dtensor5(name+'G_C')
        self._E_S = dtensor5(name+'E_S')
        self._G_S = dtensor5(name+'G_S')
        self._Reshape_C_S = dtensor5(name+'Reshape_C_S')
        self._lambda_OPL = T.dscalar(name+'lambda_OPL')
        self._w_OPL = T.dscalar(name+'lambda_OPL')
        self.state = None
    def create_filters(self):
        if self.config.get('leaky-heat-equation','0') == '0':
            self._C = conv3d(conv3d(conv3d(self._L,self._E_n_C),self._TwuTu_C),self._G_C)
            self._S = conv3d(conv3d(self._C,self._E_S),self._G_S)
            self._I_OPL = self._lambda_OPL * (conv3d(self._C,self._Reshape_C_S) - self._w_OPL * self._S)
            self.input_variables = [self._L]
            self.internal_variables = [self._E_n_C,self._TwuTu_C,self._G_C,self._E_S,self._G_S,self._Reshape_C_S, self._lambda_OPL,self._w_OPL]
            self.output_variable = self._I_OPL
            self.compute_function= theano.function(self.input_variables + self.internal_variables, self.output_variable)
            self.num_E_n_C = m_en_filter(int(self.config.get('center-n__uint',0)),float(self.config.get('center-tau__sec')),
                                      normalize=True,retina=self.retina)
            self.num_G_C = m_g_filter(float(self.config.get('center-sigma__deg')),float(self.config.get('center-sigma__deg')),retina=self.retina,normalize=True,even=False)
            raise Exception('This class is only for leaky-heat-equation != 0')
        else:
            L = conv3d(conv3d(self._L,self._E_n_C),self._TwuTu_C)[0,:,0,:,:]
            self._smoothing_a1 = T.dmatrix('smooth_a1')
            self._smoothing_a2 = T.dmatrix('smooth_a2')
            self._smoothing_a3 = T.dmatrix('smooth_a3')
            self._smoothing_a4 = T.dmatrix('smooth_a4')
            self._smoothing_b1 = T.dmatrix('smooth_b1')
            self._smoothing_b2 = T.dmatrix('smooth_b2')
            def smooth_function_forward(L,a1,a2,b1,b2,Y1,Y2,I2):
                Y0 = a1 * L + a2 * I2 + b1 * Y1 + b2 * Y2
                return [Y0, Y1, L]
            def smooth_function_backward(L,Ybuf,a3,a4,b1,b2,R,Y1,Y2,I2):
                Y0 = a3 * L + a4 * I2 + b1 * Y1 + b2 * Y2
                return [Y0+Ybuf, Y0, Y1, L]
            result_forward_x, updates = theano.scan(fn=smooth_function_forward,
                                          outputs_info = [T.ones_like(L[0,:]),T.ones_like(L[0,:]),T.ones_like(L[0,:])],
                                          sequences = [L,self._smoothing_a1,self._smoothing_a2,self._smoothing_b1,self._smoothing_b2])
            result_backward_x, updates_backward_x = theano.scan(fn=smooth_function_backward,
                                          outputs_info = [T.ones_like(L[0,:]),T.ones_like(L[0,:]),T.ones_like(L[0,:]),T.ones_like(L[0,:])],
                                          sequences = [L[::-1],result_forward_x[0][::-1],self._smoothing_a3[::-1],self._smoothing_a4[::-1],self._smoothing_b1[::-1],self._smoothing_b2[::-1]])
            result_forward_y, updates_forward_y = theano.scan(fn=smooth_function_forward,
                                          outputs_info = [T.ones_like(L[0,:]),T.ones_like(L[0,:]),T.ones_like(L[0,:])],
                                          sequences = [result_backward_x[0].transpose(),self._smoothing_a1,self._smoothing_a2,self._smoothing_b1,self._smoothing_b2])
            result_backward_y, updates_backward_y = theano.scan(fn=smooth_function_backward,
                                          outputs_info = [T.ones_like(L[0,:]),T.ones_like(L[0,:]),T.ones_like(L[0,:]),T.ones_like(L[0,:])],
                                          sequences = [result_backward_x[0].transpose()[::-1],result_forward_y[0][::-1],self._smoothing_a3[::-1],self._smoothing_a4[::-1],self._smoothing_b1[::-1],self._smoothing_b2[::-1]])
            smooth_all = theano.function(inputs=
                                                [L,self._smoothing_a1,self._smoothing_a2,self._smoothing_a3,self._smoothing_a4,self._smoothing_b1,self._smoothing_b2], 
                                                outputs=result_backward_y, updates=updates_backward_y)
            self._C = result_backward_y[0].transpose()[::-1][np.newaxis,:,np.newaxis,:,:]
            self._S = conv3d(conv3d(self._C,self._E_S),self._G_S)
            self._I_OPL = self._lambda_OPL * (conv3d(self._C,self._Reshape_C_S) - self._w_OPL * self._S)
            self.input_variables = [self._L]
            self.internal_variables = [self._E_n_C,self._TwuTu_C,self._E_S,self._G_S,self._Reshape_C_S, self._lambda_OPL,self._w_OPL,
                                        self._smoothing_a1,self._smoothing_a2,self._smoothing_a3,self._smoothing_a4,self._smoothing_b1,self._smoothing_b2]
            self.output_variable = self._I_OPL
            self.compute_function= theano.function(self.input_variables + self.internal_variables, self.output_variable)
            self.num_E_n_C = m_e_filter(float(self.config.get('center-tau__sec')),retina=self.retina,normalize=True)
            self.num_G_C = np.ones((1,1,1,1,1))
        self.num_TwuTu_C = m_t_filter(float(self.config.get('undershoot',{}).get('tau__sec')),
                                  float(self.config.get('undershoot',{}).get('relative-weight')),
                                  normalize=True,retina=self.retina,epsilon=0.01)
        self.num_E_S = m_e_filter(float(self.config.get('surround-tau__sec')),retina=self.retina,normalize=True)
        self.num_G_S = m_g_filter(float(self.config.get('surround-sigma__deg')),float(self.config.get('surround-sigma__deg')),retina=self.retina,normalize=True,even=False)
        self.num_Reshape_C_S = fake_filter(self.num_G_S,self.num_E_S)
        self.num_lambda_OPL = self.config.get('opl-amplification',0.25) / self.retina.config.get('input-luminosity-range',255.0)
        self.num_w_OPL = self.config.get('opl-relative-weight',0.7)
    def __repr__(self):
        return '[OPL Leaky Heat Eq Node] Shape: '+str(fake_filter_shape(self.num_E_n_C,self.num_TwuTu_C,self.num_G_C,self.num_E_S,self.num_G_S))+'\n -> C: '+str(self.num_G_C.shape)+' S: '+str(self.num_G_S.shape)
    def run(self,input,t_start=0):
        gCoupling = float(self.config.get('center-sigma__deg'))**2/(2*float(self.config.get('center-tau__sec')))
        sigma_0 = np.sqrt(2*gCoupling*self.retina.steps_to_seconds(1.0))
        deriche_map = retina_virtualretina.deriche_filter_density_map(self.retina, sigma0 = sigma_0, Nx = input.shape[3], Ny = input.shape[4])
        deriche_coefficients = retina_base.deriche_coefficients(deriche_map)
        all_filters_shape = fake_filter_shape(self.num_E_n_C,self.num_TwuTu_C,self.num_E_S,self.num_G_S)
        num_L = np.pad(input.copy(),[(0,0),(0,0),(0,0),(all_filters_shape[3]/2,all_filters_shape[3]/2),(all_filters_shape[4]/2,all_filters_shape[4]/2)],mode='edge')
        if self.state is not None:
            num_L = np.concatenate([self.state,num_L],1)
        else:
            num_L = np.concatenate([[num_L[0,0,:,:,:]]*(all_filters_shape[1]-1),num_L[0,:,:,:,:]],0)[np.newaxis,:,:,:,:]
        self.state = num_L[:,-(all_filters_shape[1]-1):,:,:,:]
        return self.compute_function(num_L,self.num_E_n_C,self.num_TwuTu_C,self.num_E_S,
            self.num_G_S,self.num_Reshape_C_S,self.num_lambda_OPL,self.num_w_OPL,
            deriche_coefficients['A1'],deriche_coefficients['A2'],deriche_coefficients['A3'],deriche_coefficients['A4'],
            deriche_coefficients['B1'],deriche_coefficients['B2'])
    def plot(self):
        import matplotlib.pylab as plt
        plt.figure()
        plt.suptitle(self.name)
        plt.subplot(2,2,1)
        plt.title('Center')
        plt.imshow(self.num_G_C[0,0,0,:,:],interpolation='nearest',vmin=min(np.min(self.num_G_C),0),vmax=min(np.max(self.num_G_C),1))
        plt.subplot(2,2,2)
        plt.title('Center')
        plt.plot(self.num_E_n_C[0,:,0,0,0])
        plt.plot(self.num_TwuTu_C[0,:,0,0,0])
        plt.subplot(2,2,3)
        plt.title('Surround')
        plt.imshow(self.num_G_S[0,0,0,:,:],interpolation='nearest',vmin=min(np.min(self.num_G_S),0),vmax=min(np.max(self.num_G_S),1))
        plt.subplot(2,2,4)
        plt.title('Surround')
        plt.plot(self.num_E_S[0,:,0,0,0])
        plt.tight_layout()

class VirtualRetinaBipolarLayerNode(VirtualRetinaNode):
    """

    Example Configuration::

        'contrast-gain-control': {
            'opl-amplification__Hz': 50, # for linear OPL: ampOPL = relative_ampOPL / fatherRetina->input_luminosity_range ;
                                                       # `ampInputCurrent` in virtual retina
            'bipolar-inert-leaks__Hz': 50,             # `gLeak` in virtual retina
            'adaptation-sigma__deg': 0.2,              # `sigmaSurround` in virtual retina
            'adaptation-tau__sec': 0.005,              # `tauSurround` in virtual retina
            'adaptation-feedback-amplification__Hz': 0 # `ampFeedback` in virtual retina
        },
    """
    def __init__(self,retina=None,config=None,name=None):
        self.retina = retina 
        self.config = config
        self.state = None
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
    def plot(self):
        pass
    def create_filters(self):
        pass
    def __repr__(self):
        return '[Bipolar Layer Node/contrast-gain-control beta!] Differential Equation'#+str(self.num_G_bip.shape)
    def run(self,input,t_start=0):
        from scipy.ndimage.filters import gaussian_filter
        self.input_amp = float(self.config.get('opl-amplification__Hz',100))
        self.g_leak = float(self.config.get('bipolar-inert-leaks__Hz',50))
        step_size = float(self.retina.config.get('temporal-step__sec',0.001))
        self.lambda_amp = float(self.config.get('adaptation-feedback-amplification__Hz',50))
        self.sigmaSurround = self.retina.degree_to_pixel(self.config.get('adaptation-sigma__deg',0.2))
        self.tauSurround = self.retina.steps_to_seconds(self.config.get('adaptation-tau__sec',0.005))
        self.controlCond_a,self.controlCond_b = retina_base.ab_filter_exp(self.tauSurround,self.retina.steps_to_seconds(1.0))
        input_images = input.reshape((-1,input.shape[-2],input.shape[-1]))
        size = (input_images.shape[-2],input_images.shape[-1])
        outputs = np.zeros((3,)+input_images.shape)
        storeInputs = np.zeros((2,)+size)
        controlCond_last_values = np.zeros((len(self.controlCond_a),)+size)
        controlCond_last_inputs = np.zeros((len(self.controlCond_b)+1,)+size)
        daValues = np.zeros(size)
        preceding = np.zeros(size)
        inputNernst = np.array([0.0,0])
        isInputNernst = np.array([False,True]) # is the synapse a conductance port?
        if self.state is not None:
            preceding,daValues,controlCond_last_values,controlCond_last_inputs = self.state
        for i,input_image in enumerate(input_images):
            # corresponding call in VirtualRetina: excitCells.feedCurrent(input_image,0)
            storeInputs[0] = self.input_amp * input_image
            # corresponding call in VirtualRetina: excitCells.feedConductance(controlCond,1,True)
            storeInputs[1] = controlCond_last_values[0]
            inputNernst = np.array([0.0,0])
            # corresponding call in VirtualRetina: controlCond.feedInput(excitCells)
            controlCond_last_inputs[0] = self.lambda_amp*np.abs(daValues)**2

            # corresponding call in VirtualRetina: excitCells.tempStep
            preceding, daValues = daValues, preceding
            totalCond = self.g_leak * np.ones(size)
            if np.sum(isInputNernst!=0) > 0:
                totalCond += np.sum(isInputNernst[:,np.newaxis,np.newaxis]*storeInputs,0)
            attenuationMap = np.exp(-step_size*totalCond)
            if np.sum(isInputNernst!=0) > 0:
                storeInputs[isInputNernst] = inputNernst[isInputNernst]*storeInputs[isInputNernst]
            totalInput = np.sum(storeInputs,0)/totalCond # obtaining E_infinity
            daValues = ((preceding - totalInput) * attenuationMap) + totalInput
            excitCells_daValues = daValues
            outputs[0,i] = excitCells_daValues
            outputs[1,i] = attenuationMap
            # missing feature from Virtual Retina:
            ## optinal blur


            # corresponding call in VirtualRetina: controlCond.tempStep
            ##BaseMapFilter::tempStep();
            ### if(last_inputs[0])
            ### this->spatialFiltering(last_inputs[0]);
            ### recursive filtering:
            # Advance values
            controlCond_last_values = np.concatenate([[np.zeros(size)],controlCond_last_values[:-1]])
            if len(self.controlCond_b) > 0:
                controlCond_last_values[0] += np.dot(controlCond_last_inputs[:-1].transpose(),self.controlCond_b).transpose()
            if len(self.controlCond_a) > 0:
                controlCond_last_values[0] -= np.dot(controlCond_last_values[1:].transpose(),self.controlCond_a[1:]).transpose()
            controlCond_last_values[0] /= self.controlCond_a[0]
            if self.sigmaSurround > 0.0:
                controlCond_last_values[0] = gaussian_filter(controlCond_last_values[0],self.sigmaSurround)
            # Advance inputs
            outputs[2,i] = controlCond_last_values[0]
            controlCond_last_inputs = np.concatenate([[np.zeros(size)],controlCond_last_inputs[:-1]])
            # missing feature from Virtual Retina:
            ##if(gCoupling!=0)
            ##  leakyHeatFilter.radiallyVariantBlur( *targets ); //last_values...
        self.state = preceding,daValues,controlCond_last_values,controlCond_last_inputs
        outputs = np.array(outputs)
        return outputs


class VirtualRetinaGanglionInputLayerNode(VirtualRetinaNode):
    """
    The input current to the ganglion cells is filtered through a gain function.

    :math:`I_{Gang}(x,y,t) = G * N(eT * V_{Bip})`


    :math:`N(V) = \\frac{i^0_G}{1-\lambda(V-v^0_G)/i^0_G}` (if :math:`V < v^0_G`)

    
    :math:`N(V) = i^0_G + \lambda(V-v^0_G)` (if :math:`V > v^0_G`)

        Example configuration:

            {
                'name': 'Parvocellular Off',
                'enabled': True,
                'sign': -1,
                'transient-tau__sec':0.02,
                'transient-relative-weight':0.7,
                'bipolar-linear-threshold':0,
                'value-at-linear-threshold__Hz':37,
                'bipolar-amplification__Hz':100,
                'sigma-pool__deg': 0.0,
                'spiking-channel': {
                    ...
                }
            },
            {
                'name': 'Magnocellular On',
                'enabled': False,
                'sign': 1,
                'transient-tau__sec':0.03,
                'transient-relative-weight':1.0,
                'bipolar-linear-threshold':0,
                'value-at-linear-threshold__Hz':80,
                'bipolar-amplification__Hz':400,
                'sigma-pool__deg': 0.1,
                'spiking-channel': {
                    ...
                }
            },

    """
    def __init__(self,retina=None,config=None,name=None):
        self.retina = retina 
        self.config = config
        self.state = None
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self._V_bip = dtensor5(name+"V_bip")
        self._T_G = dtensor5(name+"T_G")
        self._V_bip_E = conv3d(self._V_bip,self._T_G)
        self._i_0_G = T.dscalar(name+"i_0_G")
        self._v_0_G = T.dscalar(name+"v_0_G")
        self._lambda_G = T.dscalar(name+"lambda_G")
        self._G_gang = dtensor5(name+"G_gang")

        self._N = theano.tensor.switch(self._V_bip_E < self._v_0_G, 
                                 self._i_0_G/(1-self._lambda_G*(self._V_bip_E-self._v_0_G)/self._i_0_G),
                                 self._i_0_G + self._lambda_G*(self._V_bip_E-self._v_0_G))

        self.compute_N = theano.function([self._V_bip, self._T_G, self._i_0_G, self._v_0_G, self._lambda_G], self._N)

        self._I_Gang = conv3d(self._N,self._G_gang)

        self.compute_I_Gang = theano.function([self._V_bip, self._T_G, self._i_0_G, self._v_0_G, self._lambda_G, self._G_gang], self._I_Gang)
    def create_filters(self):
        self.num_sign = self.config.get('sign',1)
        self.num_T_G = float(self.num_sign) * m_t_filter(self.config.get('transient-tau__sec',0.04),
                                                            self.config.get('transient-relative-weight',0.75),
                                                            normalize=True,
                                                            retina=self.retina)
        self.num_i_0_G = float(self.config.get('value-at-linear-threshold__Hz',70.0))
        self.num_v_0_G = float(self.config.get('bipolar-linear-threshold',0.0))
        self.num_lambda_G = float(self.config.get('bipolar-amplification__Hz',100.0))# * 0.05
        self.num_G_gang = m_g_filter(self.config.get('sigma-pool__deg',0.0),
                                     self.config.get('sigma-pool__deg',0.0),
                                     retina=self.retina,even=False,normalize=True)
    def __repr__(self):
        return '[Ganglion Input Node] Shape: '+str(fake_filter_shape(self.num_G_gang,self.num_T_G))
    def run(self,input,t_start=0):
        num_V_bip = input.reshape((1,input.shape[0],1,input.shape[1],input.shape[2]))
        all_filters_shape = fake_filter_shape(self.num_G_gang,self.num_T_G)
        num_V_bip = np.pad(num_V_bip.copy(),[(0,0),(0,0),(0,0),(all_filters_shape[3]/2,all_filters_shape[3]/2),(all_filters_shape[4]/2,all_filters_shape[4]/2)],mode='edge')
        if all_filters_shape[1] > 1:
            if self.state is not None:
                num_V_bip = np.concatenate([self.state[:,1:,:,:,:],num_V_bip],1)
            else:
                num_V_bip = np.concatenate([[num_V_bip[0,0,:,:,:]] * (all_filters_shape[1]-1),num_V_bip[0,:,:,:,:]],0)[np.newaxis,:,:,:,:]
            # We only need to remember a state if we have more than one timepoint in our filter.
            self.state = num_V_bip[:,-all_filters_shape[1]:,:,:,:]
        if self.config.get('sigma-pool__deg',0.0) == 0.0:
            # without blurring convolution
            return self.compute_N(num_V_bip, 
                    self.num_T_G, self.num_i_0_G, self.num_v_0_G, self.num_lambda_G)#[:,all_filters_shape[1]:,:,:,:]
        # with blurring convolution
        return self.compute_I_Gang(num_V_bip, 
                    self.num_T_G, self.num_i_0_G, self.num_v_0_G, self.num_lambda_G, self.num_G_gang)#[:,all_filters_shape[1]:,:,:,:]
    def plot(self):
        import matplotlib.pylab as plt
        plt.figure()
        plt.suptitle(self.name)
        plt.subplot(2,2,1)
        plt.title('G')
        plt.imshow(self.num_G_gang[0,0,0,:,:],interpolation='nearest',vmin=min(np.min(self.num_G_gang),0),vmax=min(np.max(self.num_G_gang),1))
        plt.subplot(2,2,2)
        plt.title('T')
        plt.plot(self.num_sign * self.num_T_G[0,:,0,0,0])
        plt.tight_layout()


class VirtualRetinaGanglionSpikingLayerNode(VirtualRetinaNode):
    """
    **TODO:DONE** ~~The refractory time now working!~~

    The ganglion cells recieve the gain controlled input and produce spikes. 

    When the cell is not refractory, :math:`V` moves as:

    $$ \\\\dfrac{ dV_n }{dt} = I_{Gang}(x_n,y_n,t) - g^L V_n(t) + \eta_v(t)$$

    Otherwise it is set to 0.
    """
    def __init__(self,retina=None,config=None,name=None):
        self.retina = retina 
        self.config = config
        self.state = None
        self.last_noise_slice = None
        if name is None:
            name = str(uuid.uuid4())
        self.name = self.config.get('name',name)
        self._k_gang = T.iscalar(name+"k_gang")
        self._refrac = T.dscalar(name+"refrac")
        self._I_gang = T.dtensor3(name+"I_gang")
        self._noise_gang = T.dtensor3(name+"noise_gang") 
        self._noise_sigma = T.dscalar(name+"noise_sigma") 
        self._initial_refr = T.dmatrix(name+"initial_refr")
        self._refr_sigma = T.dscalar(name+"refr_sigma") 
        self._refr_mu = T.dscalar(name+"refr_mu") 
        self._g_L = T.dscalar(name+"g_L")
        self._tau = T.dscalar(name+"tau")
        self._V_initial = T.dmatrix(name+"V_initial")
        def spikeStep(I_gang, noise_gang,noise_gang_prev,
                      prior_V, prior_refr,  
                      noise_sigma, refr_mu, refr_sigma, g_L,tau_gang):
            V = prior_V + (I_gang - g_L * prior_V + noise_sigma*(noise_gang))*tau_gang
            V = theano.tensor.switch(T.gt(prior_refr, 0.5), 0.0, V)
            spikes = T.gt(V, 1.0)
            refr = theano.tensor.switch(spikes,
                    prior_refr + refr_mu + refr_sigma * noise_gang,
                    prior_refr - 1.0
                    )
            next_refr = theano.tensor.switch(T.lt(refr, 0.0),0.0,refr)
            return [V,next_refr]

        self._result, updates = theano.scan(fn=spikeStep,
                                      outputs_info=[self._V_initial,T.zeros_like(self._initial_refr)],
                                      sequences = [self._I_gang,dict(input=self._noise_gang, taps=[-0,-1])],
                                      non_sequences=[self._noise_sigma, self._refr_mu, self._refr_sigma, self._g_L, self._tau],
                                      n_steps=self._k_gang)

        self.compute_V_gang = theano.function(inputs=[self._I_gang,self._V_initial,self._initial_refr,self._noise_gang,
                                                      self._noise_sigma, self._refr_mu, self._refr_sigma, self._g_L,
                                                      self._tau,self._k_gang], 
                                              outputs=self._result, 
                                              updates=updates)
    def create_filters(self):
        self.num_g_L = self.config.get('g-leak__Hz',10)#*self.retina.config.get('temporal-step__sec',0.01)
        g_infini = self.num_g_L
        self.num_sigma_V = self.config.get('sigma-V',0.1)*np.sqrt(2*self.retina.seconds_to_steps(g_infini))#*self.retina.steps_to_seconds(1)*self.retina.steps_to_seconds(1))
        self.num_sigma_refr = self.retina.seconds_to_steps(self.config.get('refr-stdev__sec',0.0))
        self.num_mu_refr = self.retina.seconds_to_steps(self.config.get('refr-mean__sec',0.005))
        self.random_init = self.config.get('random-init',None)
    def __repr__(self):
        return '[Ganglion Spike Node] Differential Equation'
    def run(self,input,t_start=0):
        num_I_gang = input.reshape((input.shape[1],input.shape[3],input.shape[4]))#.repeat(4,axis=0)
        if self.state is not None:
            num_V_initial, num_initial_refr = self.state
        else:
            num_initial_refr = np.zeros((num_I_gang.shape[1],num_I_gang.shape[2]))
            if self.random_init:
                num_V_initial = np.random.rand(input.shape[3],input.shape[4])
            else:
                num_V_initial = np.zeros((input.shape[3],input.shape[4]))
        num_noise_gang = np.random.randn(*num_I_gang.shape) # gaussian noise for refr as well as sigma V
        # we use two noise values for sigma_V (taps -0 and -1), so we add the last noise image at the front
        if self.last_noise_slice is None:
            self.last_noise_slice = np.random.randn(*((1,)+num_I_gang.shape[1:]))
        num_noise_gang = np.concatenate([self.last_noise_slice,num_noise_gang],axis=0)
        self.last_noise_slice = num_noise_gang[-1:] # remember this slice for the next call
        self.num_noise_gang = num_noise_gang
        num_k_gang = num_I_gang.shape[0]
        num_tau = self.retina.config.get('temporal-step__sec',0.01)#/4.0
        out_gang,out_refr = self.compute_V_gang(num_I_gang,num_V_initial,num_initial_refr,num_noise_gang,
                            self.num_sigma_V,self.num_mu_refr,self.num_sigma_refr,
                            self.num_g_L,num_tau,num_k_gang)
        # out_gang is 3d
        # out_refr is 3d
        self.state = out_gang[-1],out_refr[-1]
        spikes = list(np.where(np.diff(out_refr,axis=0)>0.0))
        spikes[0] += t_start
        return out_gang,spikes,out_refr
    def plot(self):
        pass

class VirtualRetina(object):
    """
        This class runs the retina simulation.

        It contains a configuration (default RetinaConfiguration if not provided) and nodes that are executed sequentially.

        First :py:obj:`self.opl_node` and :py:obj:`self.bipolar_node` are executed, then each ganglion layer channel separetly.

        Visuallization can be done with :py:meth:`print_nodes()` and :py:meth:`plot_nodes()`.

        :py:obj:`pixel_per_degree` can be set directly, overriding the configuration.
    """
    def __init__(self,config=None,pixel_per_degree=None):
        self.config = config
        self.clear_output()
        if self.config is None:
            self.config = RetinaConfiguration()
        self.pixel_per_degree = self.config.get('pixels-per-degree',20.0)
        if pixel_per_degree is not None:
            self.pixel_per_degree = pixel_per_degree
        self.steps_per_second = 1.0/self.config.get('temporal-step__sec',1.0/1000.0)
        self.input_luminosity_range = self.config.get('input-luminosity-range',255.0)
        self.ganglion_channels = []
        self.opl_node = VirtualRetinaOPLLayerNode(name='OPL Layer',retina=self,config=self.config.retina_config['outer-plexiform-layers'][0])
        self.opl_node.create_filters()
        self.bipolar_node = VirtualRetinaBipolarLayerNode(name='Bipolar Layer',retina=self,config=self.config.retina_config['contrast-gain-control'])
        if self.config.get('replace_bipolar_layer_with_fake',False):
            self.bipolar_node = VirtualRetinaBipolarLayerNode_no_E(name='Bipolar Layer (no Exp)',retina=self,config=self.config.retina_config['contrast-gain-control'])
        self.bipolar_node.create_filters()
        for ganglion_config in self.config.retina_config.get('ganglion-layers',[]):
            if ganglion_config.get('enabled',True):
                gl_name = ganglion_config.get('name','')
                if gl_name != '':
                    gl_name = ': '+gl_name
                ganglion_input_node = VirtualRetinaGanglionInputLayerNode(name='Ganglion Input Layer'+gl_name,retina=self,config=ganglion_config)
                ganglion_input_node.create_filters()
                if 'spiking-channel' in ganglion_config and ganglion_config['spiking-channel'].get('enabled',True) != False:
                    ganglion_spike_node = VirtualRetinaGanglionSpikingLayerNode(name='Ganglion Spiking Layer'+gl_name,retina=self,config=ganglion_config['spiking-channel'])
                    ganglion_spike_node.create_filters()
                    self.ganglion_channels.append([ganglion_input_node,ganglion_spike_node])
                else:
                    self.ganglion_channels.append([ganglion_input_node])
    def clear_output(self):
        self.output = {}
        self.output_names = [] # we want a list, because the dict keys are unordered
        self.output_last_t = 0
    def degree_to_pixel(self,degree):
        return float(degree) * self.pixel_per_degree
    def pixel_to_degree(self,pixel):
        return float(pixel) / self.pixel_per_degree
    def seconds_to_steps(self,t):
        return float(t) * self.steps_per_second
    def steps_to_seconds(self,steps):
        return float(steps) / self.steps_per_second
    def print_nodes(self):
        for node in [self.opl_node,self.bipolar_node]:
            print repr(node)
        for channel in self.ganglion_channels:
            print 'Channel:'
            for node in channel:
                print '\t', repr(node)
            # after last layer output is returned
    def plot_nodes(self):
        for node in [self.opl_node,self.bipolar_node]:
            node.plot()
        for channel in self.ganglion_channels:
            for node in channel:
                node.plot()
    def run(self,input_images,save_output=False,print_debug=False):
        import datetime
        input = input_images
        starttime = datetime.datetime.now()
        new_output = {}
        new_output_names = [] # we want a list, because the dict keys are unordered
        for node in [self.opl_node,self.bipolar_node]:
            if print_debug:
                print '[',datetime.datetime.now()-starttime,']', node
                print '>> Input is:',str(input.shape),'mean: ',np.nanmean(input),np.mean(input)
            input = node.run(input,t_start=self.output_last_t)
            if save_output or node.config.get('save_output',False):
                new_output[find_nonconflict(node.name,new_output_names)] = input
        bipolar_output = input.copy()
        outputs = []
        for channel in self.ganglion_channels:
            input = bipolar_output[0,:,:,:].copy()
            for node in channel:
                if print_debug:
                    print '[',datetime.datetime.now()-starttime,']', node
                    print '>> Input is:',str(input.shape),'mean: ',np.nanmean(input),np.mean(input)
                input = node.run(input,t_start=self.output_last_t)
                if print_debug:
                    if type(input) is tuple:
                        print '>> Output is:',str(input[0].shape),'mean: ',np.nanmean(input[0]),np.mean(input[0])
                    else:
                        print '>> Output is:',str(input.shape),'mean: ',np.nanmean(input),np.mean(input)
                if save_output or node.config.get('save_output',False):
                    new_output[find_nonconflict(node.name,new_output_names)] = input
            # after last layer output is returned
            if print_debug:
                if len(input) == 2:
                    print '>> Output[0] is:',str(input[0].shape),'mean: ',np.nanmean(input[0]),np.mean(input[0])
                    print '>> Output[1] is:',str(len(input[1]))
                else:
                    print '>> Output[0] is:',str(input[0].shape),'mean: ',np.nanmean(input[0]),np.mean(input[0])
            outputs.append(input)
        if len(new_output_names) > 0:
            if print_debug:
                print "Outputs were saved for nodes: "+(", ".join(new_output_names))
                print "Access or remove them with the .output[..] attribute."
            for n in new_output_names:
                if not n in self.output:
                    self.output[n] = new_output[n]
                else:
                    self.output[n] = retina_base.concatenate_time(self.output[n],new_output[n])
            self.output_names = new_output_names # the order is always from the last call to run. In any case they should be identical for each call.
            self.output_last_t += input_images.shape[1]
        return outputs
    def plot_outputs(self, input_images,ganglion_channel=0, s_1=slice(None,None,None),s_2=slice(None,None,None),t=slice(None,None,None),figsize=(7,7)):
        import matplotlib.pylab as plt
        if type(ganglion_channel) is int:
            ganglion_channel = self.ganglion_channels[ganglion_channel][0].name
        _o = [r.reshape((-1,r.shape[-2],r.shape[-1])) if len(r.shape) != 4 else r for r in [self.output['OPL Layer'],
                                                                self.output['Bipolar Layer'],
                                                                self.output[ganglion_channel],
                                                                self.output['Ganglion Spiking Layer: '+ganglion_channel][0]]]
        fig = plt.figure(figsize=figsize)
        def xlim(t):
            # either use the complete stimulus time or the t slice for limiting the x axis
            plt.xlim(0 if t.start is None else t.start,len(input_images) if t.stop is None else t.stop)
        plt.subplot(711)
        plt.title('Input')
        plt.plot(range(len(input_images))[t],np.mean(input_images[t,s_1,s_2],(1,2)),color='black')
        xlim(t)
        plt.subplot(712)
        plt.title('OPL Output')
        plt.plot(range(len(input_images))[-len(_o[0]):][t],np.mean(_o[0][t,s_1,s_2],(1,2)),color='orange')
        xlim(t)
        plt.subplot(713)
        plt.title('Bipolar Layer Output')
        plt.plot(range(len(input_images))[-len(_o[1][0]):][t],np.mean(_o[1][0][t,s_1,s_2],(1,2)),color='lightgreen')
        xlim(t)
        plt.subplot(714)
        plt.title('Bipolar Layer Output (attenuationMap)')
        plt.plot(range(len(input_images))[-len(_o[1][1]):][t],np.mean(_o[1][1][t,s_1,s_2],(1,2)),color='blue')
        xlim(t)
        plt.subplot(715)
        plt.title('Bipolar Layer Output (Surround)')
        plt.plot(range(len(input_images))[-len(_o[1][2]):][t],np.mean(_o[1][2][t,s_1,s_2],(1,2)),color='orange')
        xlim(t)
        plt.subplot(716)
        plt.title('Ganglion Input '+ganglion_channel)
        plt.plot(range(len(input_images))[-len(_o[2]):][t],np.mean(_o[2][t,s_1,s_2],(1,2)))
        xlim(t)
        plt.subplot(717)
        plt.title('Ganglion Spikes '+ganglion_channel)
        plt.plot(range(len(input_images))[-len(_o[3]):][t],np.mean(_o[3][t,s_1,s_2],(1,2)))
        xlim(t)
        plt.tight_layout()
        return fig
    def plot_outputs_2(self,ganglion_channel=0,figsize=(16,3),t=slice(0,None)):
        import matplotlib.pylab as plt
        if type(ganglion_channel) is int:
            ganglion_channel = self.ganglion_channels[ganglion_channel][0].name
        N = self.output['OPL Layer'].shape[-1]
        colors = plt.cm.Accent(np.linspace(0.1,0.9,N))
        plt.figure(figsize=figsize)
        plt.suptitle(ganglion_channel)
        plt.subplot(151)
        plt.gca().set_color_cycle(colors)
        plt.plot(self.output['OPL Layer'].reshape((self.output['OPL Layer'].shape[1],-1))[:,(4*N):(5*N)])
        plt.xlabel('time')
        plt.title('OPL Layer')
        plt.subplot(152)
        plt.gca().set_color_cycle(colors)
        plt.plot(self.output['Bipolar Layer'][0].reshape((self.output['Bipolar Layer'].shape[1],-1))[:,(4*N):(5*N)])
        plt.xlabel('time')
        plt.title('Bipolar Layer')
        plt.subplot(153)
        plt.gca().set_color_cycle(colors)
        plt.plot(self.output[ganglion_channel].reshape((self.output[ganglion_channel].shape[1],-1))[:,(4*N):(5*N)])
        plt.xlabel('time')
        plt.title(ganglion_channel+' Ganglion Input')
        plt.subplot(154)
        plt.gca().set_color_cycle(colors)
        gv = self.output['Ganglion Spiking Layer: '+ganglion_channel][0]
        plt.plot(gv.reshape((gv.shape[0],-1))[:,(4*N):(5*N)])
        plt.xlim(t.start,t.stop)
        plt.xlabel('time')
        plt.title('Membrane Potential: '+ganglion_channel)
        plt.subplot(155)
        plt.gca().set_color_cycle(colors)
        spikes = self.output['Ganglion Spiking Layer: '+ganglion_channel][1]
        for i in range((4*N),(5*N)):
            plt.plot(spikes[0][spikes[1]+N*spikes[2]==i],[i]*np.sum(spikes[1]+N*spikes[2]==i),'|')
        #xlim(600,800)
        plt.xlabel('time')
        plt.title('Ganglion Spikes: '+ganglion_channel)
        plt.tight_layout()


class VirtualRetinaBeta(VirtualRetina):
    """
        This class runs the retina simulation.

        It contains a configuration (default RetinaConfiguration if not provided) and nodes that are executed sequentially.

        First :py:obj:`self.opl_node` and :py:obj:`self.bipolar_node` are executed, then each ganglion layer channel separetly.

        Visuallization can be done with :py:meth:`print_nodes()` and :py:meth:`plot_nodes()`.

        :py:obj:`pixel_per_degree` can be set directly, overriding the configuration.
    """
    def __init__(self,config=None,pixel_per_degree=None):
        self.config = config
        self.clear_output()
        if self.config is None:
            self.config = RetinaConfiguration()
        self.pixel_per_degree = self.config.get('pixels-per-degree',20.0)
        if pixel_per_degree is not None:
            self.pixel_per_degree = pixel_per_degree
        self.steps_per_second = 1.0/self.config.get('temporal-step__sec',1.0/1000.0)
        self.input_luminosity_range = self.config.get('input-luminosity-range',255.0)
        self.ganglion_channels = []
        if self.config.retina_config['outer-plexiform-layers'][0].get('leaky-heat-equation','0') != '0':
            self.opl_node = VirtualRetinaOPLLayerNodeLeakyHeat(name='OPL Layer',retina=self,config=self.config.retina_config['outer-plexiform-layers'][0])
        else:
            self.opl_node = VirtualRetinaOPLLayerNode(name='OPL Layer',retina=self,config=self.config.retina_config['outer-plexiform-layers'][0])
        self.opl_node.create_filters()

        # this is a beta component:
        self.bipolar_node = VirtualRetinaBipolarLayerNode(name='Bipolar Layer',retina=self,config=self.config.retina_config['contrast-gain-control'])

        self.bipolar_node.create_filters()
        for ganglion_config in self.config.retina_config.get('ganglion-layers',[]):
            if ganglion_config.get('enabled',True):
                gl_name = ganglion_config.get('name','')
                if gl_name != '':
                    gl_name = ': '+gl_name
                ganglion_input_node = VirtualRetinaGanglionInputLayerNode(name='Ganglion Input Layer'+gl_name,retina=self,config=ganglion_config)
                ganglion_input_node.create_filters()
                if 'spiking-channel' in ganglion_config and ganglion_config['spiking-channel'].get('enabled',True) != False:
                    ganglion_spike_node = VirtualRetinaGanglionSpikingLayerNode(name='Ganglion Spiking Layer'+gl_name,retina=self,config=ganglion_config['spiking-channel'])
                    ganglion_spike_node.create_filters()
                    self.ganglion_channels.append([ganglion_input_node,ganglion_spike_node])
                else:
                    self.ganglion_channels.append([ganglion_input_node])
