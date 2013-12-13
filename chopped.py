from pylearn2.models.mlp import Layer, MLP
from pylearn2.space import VectorSpace

class PretrainedLayer(Layer):
    """
    A layer whose weights are initialized, and optionally fixed,
    based on prior training.
    """

    def __init__(self, layer_name, layer_content, freeze_params=False, last_layer = -1):
        """
        layer_content: A Model that implements "fprop"
        freeze_params: If True, regard layer_conent's parameters as fixed
            If False, they become parameters of this layer and can be
            fine-tuned to optimize the MLP's cost function.
	    last_layer: Number of the last layer to include
        """
        self.__dict__.update(locals())
        self.layer_content.layers = self.layer_content.layers[:last_layer]
        del self.self

    def set_input_space(self, space):
        assert self.get_input_space() == space

    def get_params(self):
        if self.freeze_params:
            return []
        return self.layer_content.get_params()

    def get_input_space(self):
        return self.layer_content.get_input_space()

    def get_output_space(self):
        return self.layer_content.get_output_space()

    def fprop(self, state_below):
        return self.layer_content.fprop(state_below)

class NestedMLP(MLP):
    """
    A multilayer perceptron.
    Note that it's possible for an entire MLP to be a single
    layer of a larger MLP.
    """

    def __init__(self, layers, layer_range, model_layer_index =0,
                batch_size=None, input_space=None,
                 nvis=None, seed=None):
        """
        Instantiate an MLP.

        Parameters
        ----------
        layers : list
            A list of Layer objects. The final layer specifies
            the output space of this MLP.

        batch_size : int, optional
            If not specified then must be a positive integer.
            Mostly useful if one of your layers involves a
            Theano op like convolution that requires a hard-coded
            batch size.

        nvis : int, optional
            Number of "visible units" (input units). Equivalent
            to specifying `input_space=VectorSpace(dim=nvis)`.

        input_space : Space object, optional
            A Space specifying the kind of input the MLP accepts.
            If None, input space is specified by nvis.
        """

        if seed is None:
            seed = [2013, 1, 4]

        self.seed = seed
        self.setup_rng()

        #unravel
        layers[model_layer_index] = layers[model_layer_index].\
                layers[layer_range[0]: layer_range[1]]
        layers_ = []
        for item in layers:
            if isinstance(item, list):
                for l in item:
                    l.mlp = None
                    layers_.append(l)
            else:
                layers_.append(item)
        layers = layers_
        del layers_
        print layers[0].get_params()[0].get_value().mean()
        print layers[0].get_params()[1].get_value().mean()
        print 'an0'

        assert isinstance(layers, list)
        assert all(isinstance(layer, Layer) for layer in layers)
        assert len(layers) >= 1
        self.layer_names = set()
        for layer in layers:
            assert layer.get_mlp() is None
            if layer.layer_name in self.layer_names:
                raise ValueError("MLP.__init__ given two or more layers "
                                 "with same name: " + layer.layer_name)
            layer.set_mlp(self)
            self.layer_names.add(layer.layer_name)

        self.layers = layers
        self.batch_size = batch_size
        self.force_batch_size = batch_size

        assert input_space is not None or nvis is not None
        if nvis is not None:
            input_space = VectorSpace(nvis)

        self.input_space = input_space

        self._update_layer_input_spaces(start = layer_range[1])

        self.freeze_set = set([])

        def f(x):
            if x is None:
                return None
            return 1. / x

    def _update_layer_input_spaces(self, start):
        """
            Tells each layer what its input space should be.
            Note: this usually resets the layer's parameters!
        """
        layers = self.layers
        #layers[0].set_input_space(self.input_space)
        for i in xrange(start,len(layers)):
            layers[i].set_input_space(layers[i-1].get_output_space())


