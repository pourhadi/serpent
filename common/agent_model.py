class AgentModel:
    
    def __init__(self, frame_shape, game_inputs, model_name):
        self.frame_shape = frame_shape
        self.game_inputs = game_inputs
        self.model_name = model_name

    def restore(self, directory):
        pass
        
    def observe(self, reward, terminal):
        pass
        
    def generate_action(self, frame_buffer):
        pass
