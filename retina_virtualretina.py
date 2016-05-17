"""

This module is a compatibility layer between the Virtual Retina configurations and behaviour and the python implementation.

"""


# This dict contains tag and attribute names used in virtual retina configuration files
valid_retina_tags = {
    'retina': ['temporal-step__sec','input-luminosity-range','pixels-per-degree'],
    'basic-microsaccade-generator': ['pixels-per-degree',
                                'temporal-step__sec',
                                'angular-noise__pi-radians',
                                'period-mean__sec',
                                'period-stdev__sec',
                                'amplitude-mean__deg',
                                'amplitude-stdev__deg',
                                'saccade-duration-mean__sec',
                                'saccade-duration-stdev__sec'],
    'outer-plexiform-layer': [],
    'linear-version': ['center-sigma__deg',
                                'surround-sigma__deg',
                                'center-tau__sec',
                                'surround-tau__sec',
                                'opl-amplification',
                                'opl-relative-weight',
                                'leaky-heat-equation'],
    'undershoot': ['relative-weight','tau__sec'],
    'contrast-gain-control': ['opl-amplification__Hz',
                                'bipolar-inert-leaks__Hz',
                                'adaptation-sigma__deg',
                                'adaptation-tau__sec',
                                'adaptation-feedback-amplification__Hz'],
    'ganglion-layer': ['sign',
                                'transient-tau__sec',
                                'transient-relative-weight',
                                'bipolar-linear-threshold',
                                'value-at-linear-threshold__Hz',
                                'bipolar-amplification__Hz',
                                'sigma-pool__deg'],
    'spiking-channel': ['g-leak__Hz',
                                'sigma-V',
                                'refr-mean__sec',
                                'refr-stdev__sec',
                                'random-init'],
    'square-array': ['size-x__deg', 'size-y__deg', 'uniform-density__inv-deg'],
    'circular-array': ['fovea-density__inv-deg','diameter__deg']
}


class RetinaConfiguration:
    """
        A configuration object that writes an xml file for VirtualRetina.

        (When this is altered, silver.glue.RetinaConfiguration has to also be updated by hand)

        Does not currently care to parse an xml file, but can save/load in json instead.
        The defaults are equal to `human.parvo.xml`.

        Values can be changed either directly in the configuration dictionary, or with the `set` helperfunction::

            config = silver.glue.RetinaConfiguration()
            config.retina_config['retina']['input-luminosity-range'] = 200
            config.set('basic-microsaccade-generator.enabled') = True
            config.set('ganglion-layers.*.spiking-channel.sigma-V') = 0.5 # for all layers

    """
    def __init__(self):
        self.default()
    def default(self):
        """
        Generates a default config::

            self.retina_config = {
                        'basic-microsaccade-generator' :{
                            'enabled': False,
                            'pixels-per-degree':200,
                            'temporal-step__sec':0.005,
                            'angular-noise__pi-radians':0.3,
                            'period-mean__sec':0.2,
                            'period-stdev__sec':0,
                            'amplitude-mean__deg':0.5,
                            'amplitude-stdev__deg':0.1,
                            'saccade-duration-mean__sec':0.025,
                            'saccade-duration-stdev__sec':0.005,
                        },
                        'retina': {
                            'temporal-step__sec':0.01,
                            'input-luminosity-range':255,
                            'pixels-per-degree':5.0
                        },
                        'log-polar-scheme' : {
                            'enabled': False,
                            'fovea-radius__deg': 1.0,
                            'scaling-factor-outside-fovea__inv-deg': 1.0
                        },
                        'outer-plexiform-layers': [
                            {   
                                'version': 'linear-version',
                                'center-sigma__deg': 0.05,
                                'surround-sigma__deg': 0.15,
                                'center-tau__sec': 0.01,
                                'surround-tau__sec': 0.004,
                                'opl-amplification': 10,
                                'opl-relative-weight': 1,
                                'leaky-heat-equation': 1,
                                'undershoot': {
                                    'enabled': True,
                                    'relative-weight': 0.8,
                                    'tau__sec': 0.1
                                }
                            }
                        ],
                        'contrast-gain-control': {
                            'opl-amplification__Hz': 50, # for linear OPL: ampOPL = relative_ampOPL / fatherRetina->input_luminosity_range ;
                            'bipolar-inert-leaks__Hz': 50,
                            'adaptation-sigma__deg': 0.2,
                            'adaptation-tau__sec': 0.005,
                            'adaptation-feedback-amplification__Hz': 0
                        },
                        'ganglion-layers': [
                            {
                                'name': 'Parvocellular On',
                                'enabled': True,
                                'sign': 1,
                                'transient-tau__sec':0.02,
                                'transient-relative-weight':0.7,
                                'bipolar-linear-threshold':0,
                                'value-at-linear-threshold__Hz':37,
                                'bipolar-amplification__Hz':100,
                                'sigma-pool__deg': 0.0,
                                'spiking-channel': {
                                    'g-leak__Hz': 50,
                                    'sigma-V': 0.1,
                                    'refr-mean__sec': 0.003,
                                    'refr-stdev__sec': 0,
                                    'random-init': 1,
                                    'square-array': {
                                        'size-x__deg': 4,
                                        'size-y__deg': 4,
                                        'uniform-density__inv-deg': 20
                                    }
                                }
                            },
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
                                    'g-leak__Hz': 50,
                                    'sigma-V': 0.1,
                                    'refr-mean__sec': 0.003,
                                    'refr-stdev__sec': 0,
                                    'random-init': 1,
                                    'square-array': {
                                        'size-x__deg': 4,
                                        'size-y__deg': 4,
                                        'uniform-density__inv-deg': 20
                                    }
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
                                    'g-leak__Hz': 50,
                                    'sigma-V': 0,
                                    'refr-mean__sec': 0.003,
                                    'refr-stdev__sec': 0,
                                    'random-init': 1,
                                    'circular-array': {
                                        'fovea-density__inv-deg': 15.0
                                    }
                                }
                            },
                            {
                                'name': 'Magnocellular Off',
                                'enabled': False,
                                'sign': -1,
                                'transient-tau__sec':0.03,
                                'transient-relative-weight':1.0,
                                'bipolar-linear-threshold':0,
                                'value-at-linear-threshold__Hz':80,
                                'bipolar-amplification__Hz':400,
                                'sigma-pool__deg': 0.1,
                                'spiking-channel': {
                                    'g-leak__Hz': 50,
                                    'sigma-V': 0,
                                    'refr-mean__sec': 0.003,
                                    'refr-stdev__sec': 0,
                                    'random-init': 1,
                                    'circular-array': {
                                        'fovea-density__inv-deg': 15.0
                                    }
                                }
                            }
                        ]
                    }
        """
        self.retina_config = {
            'basic-microsaccade-generator' :{
                'enabled': False,
                'pixels-per-degree':200,
                'temporal-step__sec':0.005,
                'angular-noise__pi-radians':0.3,
                'period-mean__sec':0.2,
                'period-stdev__sec':0,
                'amplitude-mean__deg':0.5,
                'amplitude-stdev__deg':0.1,
                'saccade-duration-mean__sec':0.025,
                'saccade-duration-stdev__sec':0.005,
            },
            'retina': {
                'temporal-step__sec':0.01,
                'input-luminosity-range':255,
                'pixels-per-degree':5.0
            },
            'log-polar-scheme' : {
                'enabled': False,
                'fovea-radius__deg': 1.0,
                'scaling-factor-outside-fovea__inv-deg': 1.0
            },
            'outer-plexiform-layers': [
                {   
                    'version': 'linear-version',
                    'center-sigma__deg': 0.05,
                    'surround-sigma__deg': 0.15,
                    'center-tau__sec': 0.01,
                    'center-n__uint': 0,
                    'surround-tau__sec': 0.004,
                    'opl-amplification': 10,
                    'opl-relative-weight': 1,
                    'leaky-heat-equation': 1,
                    'undershoot': {
                        'enabled': True,
                        'relative-weight': 0.8,
                        'tau__sec': 0.1
                    }
                }
            ],
            'contrast-gain-control': {
                'opl-amplification__Hz': 50, # for linear OPL: ampOPL = relative_ampOPL / fatherRetina->input_luminosity_range ;
                'bipolar-inert-leaks__Hz': 50,
                'adaptation-sigma__deg': 0.2,
                'adaptation-tau__sec': 0.005,
                'adaptation-feedback-amplification__Hz': 0
            },
            'ganglion-layers': [
                {
                    'name': 'Parvocellular On',
                    'enabled': True,
                    'sign': 1,
                    'transient-tau__sec':0.02,
                    'transient-relative-weight':0.7,
                    'bipolar-linear-threshold':0,
                    'value-at-linear-threshold__Hz':37,
                    'bipolar-amplification__Hz':100,
                    'sigma-pool__deg': 0.0,
                    'spiking-channel': {
                        'g-leak__Hz': 50,
                        'sigma-V': 0.1,
                        'refr-mean__sec': 0.003,
                        'refr-stdev__sec': 0,
                        'random-init': 0,
                        'square-array': {
                            'size-x__deg': 4,
                            'size-y__deg': 4,
                            'uniform-density__inv-deg': 20
                        }
                    }
                },
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
                        'g-leak__Hz': 50,
                        'sigma-V': 0.1,
                        'refr-mean__sec': 0.003,
                        'refr-stdev__sec': 0,
                        'random-init': 0,
                        'square-array': {
                            'size-x__deg': 4,
                            'size-y__deg': 4,
                            'uniform-density__inv-deg': 20
                        }
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
                        'g-leak__Hz': 50,
                        'sigma-V': 0,
                        'refr-mean__sec': 0.003,
                        'refr-stdev__sec': 0,
                        'random-init': 1,
                        'circular-array': {
                            'fovea-density__inv-deg': 15.0
                        }
                    }
                },
                {
                    'name': 'Magnocellular Off',
                    'enabled': False,
                    'sign': -1,
                    'transient-tau__sec':0.03,
                    'transient-relative-weight':1.0,
                    'bipolar-linear-threshold':0,
                    'value-at-linear-threshold__Hz':80,
                    'bipolar-amplification__Hz':400,
                    'sigma-pool__deg': 0.1,
                    'spiking-channel': {
                        'g-leak__Hz': 50,
                        'sigma-V': 0,
                        'refr-mean__sec': 0.003,
                        'refr-stdev__sec': 0,
                        'random-init': 1,
                        'circular-array': {
                            'fovea-density__inv-deg': 15.0
                        }
                    }
                }
            ]
        }
    def get(self,key,default):
        """
            Retrieves values from the configuration.

            At some point this will be updated to be as powerfull as :py:meth:`set`.
        """
        return self.retina_config.get(key,default)
    def set(self,key,value,layer_filter=None):
        """
            shortcuts for frequent configuration values

            Knows where to put:

                'pixels-per-degree', 'size__deg' (if x and y are equal), 'uniform-density__inv-deg'
                all attributes of linear-version
                all attributes of undershoot

            Understands dot notation::

                conf = silver.glue.RetinaConfiguration()
                conf.set("ganglion-layers.2.enabled",True)
                conf.set("ganglion-layers.*.spiking-channel.sigma-V",0.101) # changes the value for all layers

            But whole sub-tree dicitonaries can be set as well (they replace, not update)::

                conf.set('contrast-gain-control', {'opl-amplification__Hz': 50,
                                                    'bipolar-inert-leaks__Hz': 50,
                                                    'adaptation-sigma__deg': 0.2,
                                                    'adaptation-tau__sec': 0.005,
                                                    'adaptation-feedback-amplification__Hz': 0
                                                })

            New dictionary keys are created automatically, new list elements can be created like this::

                conf.set("ganglion-layers.=.enabled",True) # copies all values from the last element
                conf.set("ganglion-layers.=1.enabled",True) # copies all values from list element 1
                conf.set("ganglion-layers.+.enabled",True) # creates a new (empty) dictionary which is probably underspecified

                conf.set("ganglion-layers.+",{
                            'name': 'Parvocellular On',
                            'enabled': True,
                            'sign': 1,
                            'transient-tau__sec':0.02,
                            'transient-relative-weight':0.7,
                            'bipolar-linear-threshold':0,
                            'value-at-linear-threshold__Hz':37,
                            'bipolar-amplification__Hz':100,
                            'spiking-channel': {
                                'g-leak__Hz': 50,
                                'sigma-V': 0.1,
                                'refr-mean__sec': 0.003,
                                'refr-stdev__sec': 0,
                                'random-init': 0,
                                'square-array': {
                                    'size-x__deg': 4,
                                    'size-y__deg': 4,
                                    'uniform-density__inv-deg': 20
                                }
                            }
                        }) # ganglion cell layer creates a new dicitonary

        """
        if key == 'pixels-per-degree':
            self.retina_config['retina']['pixels-per-degree'] = value
            self.retina_config['basic-microsaccade-generator']['pixels-per-degree'] = value
        elif key in valid_retina_tags['linear-version']:
            self.retina_config['outer-plexiform-layer']['linear-version'][key] = value
        elif key in valid_retina_tags['undershoot']:
            self.retina_config['outer-plexiform-layer']['linear-version']['undershoot'][key] = value
        elif key == 'size__deg':
            for l in self.retina_config.get('ganglion-layers',[]):
                if layer_filter is not None and not layer_filter in l['name']:
                    continue
                l['spiking-channel'] = l.get('spiking-channel',{})
                l['spiking-channel']['square-array'] = l['spiking-channel'].get('square-array',{})
                l['spiking-channel']['square-array']['enabled'] = True
                l['spiking-channel']['square-array']['size-x__deg'] = value
                l['spiking-channel']['square-array']['size-y__deg'] = value
        elif key == 'uniform-density__inv-deg':
            for l in self.retina_config.get('ganglion-layers',[]):
                if layer_filter is not None and not layer_filter in l['name']:
                    continue
                l['spiking-channel'] = l.get('spiking-channel',{})
                l['spiking-channel']['square-array'] = l['spiking-channel'].get('square-array',{})
                l['spiking-channel']['square-array']['enabled'] = True
                l['spiking-channel']['square-array']['uniform-density__inv-deg'] = value
        elif key == 'enabled':
            for l in self.retina_config.get('ganglion-layers',[]):
                if layer_filter is not None and not layer_filter in l['name']:
                    continue
                l['enabled'] = value
        else:
            # Shortcut dot notation
            def put(d, keys, item):
                if "." in keys:
                    key, rest = keys.split(".", 1)
                    if type(d) is list:
                        if key == "+":
                            d.append({})
                            put(d[-1], rest, item)
                        elif key == "=":
                            d.append(d[-1])
                            put(d[-1], rest, item)
                        elif key.startswith("="):
                            d.append(d[int(key[1:])]) # use the referenced element
                            put(d[-1], rest, item)
                        elif key == "*":
                            for i in range(len(d)):
                                put(d[i], rest, item)
                        else:
                            while int(key) >= len(d):
                                d.append({})
                            put(d[int(key)], rest, item)
                    else:
                        if key == "*":
                            for k in d.keys():
                                put(d[k], rest, item)
                        else:
                            if key not in d:
                                d[key] = {}
                            put(d[key], rest, item)
                else:
                    if type(d) is list:
                        if key == "+":
                            d.append({})
                            d[-1] = item
                        else:
                            while int(key) >= len(d):
                                d.append({})
                            d[int(key)] = item
                    else:
                        d[keys] = item
            put(self.retina_config,key,value)
    def read_json(self,filename):
        """
            Reads a full retina config json file.
        """
        self.retina_config = json.load(filename)
    def read_xml(self,filename):
        """
            Reading a full retina config xml file: Not implemented yet.
        """
        raise Exception("Not yet implemented.")
    def write_json(self,filename):
        """
            Writes a retina config json file.
        """
        json.dump(self.retina_config,filename)
    def write_xml(self,filename):
        """
            Writes a full retina config xml file.
        """
        def add_element(tag_name,parent,config,config_is_parent_config=True):
            if parent is None:
                return
            if config_is_parent_config:
                config = config.get(tag_name,{'enabled':False})
            if 'enabled' not in config or config['enabled']:
                e = ET.SubElement(parent, tag_name)
                for k in config.keys():
                    if k in valid_retina_tags[tag_name]:
                        e.set(k,str(config[k]))
                return e
        self.tree = ET.Element('retina-description-file')
        add_element('basic-microsaccade-generator', self.tree, self.retina_config)
        retina = add_element('retina', self.tree, self.retina_config)
        add_element('log-polar-scheme', retina, self.retina_config)
        for layer_config in self.retina_config.get('outer-plexiform-layers',[]):
            opl_layer = add_element('outer-plexiform-layer', retina, layer_config,False)
            lin = add_element('linear-version', opl_layer, layer_config,False)
            undershoot = add_element('undershoot', lin, layer_config)
        add_element('contrast-gain-control', retina, self.retina_config)
        for layer_config in self.retina_config.get('ganglion-layers',[]):
            ganglion_layer = add_element('ganglion-layer', retina, layer_config,False)
            spiking_channel = add_element('spiking-channel', ganglion_layer, layer_config)
            square_array = add_element('square-array', spiking_channel, layer_config.get('spiking-channel',{'enabled':False}))
            circular_array = add_element('circular-array', spiking_channel, layer_config.get('spiking-channel',{'enabled':False}))
        with open(filename,'w') as f:
            f.write(ET.tostring(self.tree))


def deriche_filter_density_map(retina, sigma0 = 1.0, Nx = None, Ny = None):
    """
        Returns a map of how strongly a point is to be blurred.

        Relevant config options of retina::

            'log-polar-scheme' : {
                'enabled': True,
                'fovea-radius__deg': 1.0,
                'scaling-factor-outside-fovea__inv-deg': 1.0
            }

        or for a circular (constant) scheme::

            'log-polar-scheme' : {
                'enabled': False,
                'fovea-radius__deg': 1.0,
                'scaling-factor-outside-fovea__inv-deg': 1.0
            } 

        The output should be used with `retina_base.deriche_coefficients` to generate the coefficient maps for a Deriche filter.
    """
    import numpy as np
    Ny = Nx if Ny is None else Ny
    x, y = np.meshgrid(np.arange(Nx),np.arange(Ny))
    midx = midy = Nx/2.0
    ratiopix = retina.degree_to_pixel(1.0)
    r = np.sqrt((x-midx)**2 + (y-midy)**2) + 0.001
    density = np.ones_like(r)
    log_polar_config = retina.config.retina_config.get('log-polar-scheme',{})
    if log_polar_config.get('enabled',False):
        log_polar_K = log_polar_config.get('scaling-factor-outside-fovea__inv-deg', 1.0)
        log_polar_R0 = log_polar_config.get('fovea-radius__deg', 1.0)
        if log_polar_K is None or log_polar_K < 0.0:
            log_polar_K = 1.0/log_polar_R0
        density = r
        density[r>log_polar_R0] = log_polar_R0 + log(1+log_polar_K*(density[r>log_polar_R0]-log_polar_R0))/log_polar_K
    return density/(sigma0*ratiopix)