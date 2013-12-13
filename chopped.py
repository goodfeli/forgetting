from pylearn2.models.mlp import Layer

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


