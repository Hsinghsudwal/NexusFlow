import tensorflow as tf

class TFModelServer:
    def __init__(self, model, config):
        self.model = tf.saved_model.load(model.path)
        self.config = config
        
    def serve(self, input_data):
        return self.model.signatures['serving_default'](tf.constant(input_data))

class ServerFactory:
    @staticmethod
    def create_server(model, config):
        if config['server_type'] == 'tensorflow':
            return TFModelServer(model, config)
        elif config['server_type'] == 'torch':
            return TorchServer(model, config)
        elif config['server_type'] == 'custom':
            return CustomServer(model, config)