from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt
import tensorflow as tf


class EncodeProcessDecode(snt.Module):
    """Full encode-process-decode model.

  The model we explore includes three components:
  - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
  - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
  - A "Decoder" graph net, which independently decodes the edge, node, and
    global attributes (does not compute relations etc.), on each message-passing
    step.

                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*
  """

    def __init__(self,
                 make_encoder_edge_model=None,
                 make_encoder_node_model=None,
                 make_encoder_global_model=None,
                 make_core_edge_model=None,
                 make_core_node_model=None,
                 make_core_global_model=None,
                 num_processing_steps=None,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 node_output_fn=None,
                 name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        self._encoder = modules.GraphIndependent(
            edge_model_fn=make_encoder_edge_model,
            node_model_fn=make_encoder_node_model,
            global_model_fn=make_encoder_global_model)
        self._core = modules.GraphNetwork(
            make_core_edge_model,
            make_core_node_model,
            make_core_global_model)
        self._decoder = modules.GraphIndependent(
            edge_model_fn=make_encoder_edge_model,
            node_model_fn=make_encoder_node_model,
            global_model_fn=make_encoder_global_model)
        self.num_processing_steps = num_processing_steps
        # Transforms the outputs into the appropriate shapes.
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
        if node_output_fn is None:
            if node_output_size is None:
                node_fn = None
            else:
                node_fn = lambda: snt.Linear(node_output_size, name="node_output")
        else:
            node_fn = node_output_fn
        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(global_output_size, name="global_output")
        self._output_transform = modules.GraphIndependent(
            edge_fn, node_fn, global_fn)

    def __call__(self, input_op):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(self.num_processing_steps):
            core_input = utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))
        return output_ops


def make_mlp(layers):
    return snt.Sequential([
        snt.nets.MLP(layers, activate_final=True),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    ])


def snt_mlp(layers):
    return lambda: make_mlp(layers)

def create_node_output_label():
    return snt.nets.MLP([2],
                        activation=tf.nn.softmax,
                        activate_final=True,
                        name="node_output")
