import hashlib
import numpy as np

class ModelWatermarker:
    def __init__(self, secret_key):
        self.secret = hashlib.sha256(secret_key.encode()).digest()
        
    def apply_watermark(self, model):
        watermark_vector = np.frombuffer(self.secret[:16], dtype=np.float32)
        for param in model.parameters():
            param.data += watermark_vector * 0.01
        return model
    
    def verify_watermark(self, model):
        signature = []
        for param in model.parameters():
            signature.append(param.data.cpu().numpy().tobytes())
        return self.secret in b''.join(signature)