import torch
import tensorflow as tf
from torch.nn import KLDivLoss

class ModelDistiller:
    def __init__(self, teacher_model, student_model, temperature=3):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.kl_div = KLDivLoss()

    def distill(self, train_loader, epochs=10):
        self.teacher.eval()
        self.student.train()
        
        for epoch in range(epochs):
            for data, _ in train_loader:
                teacher_probs = torch.softmax(self.teacher(data)/self.temperature, dim=1)
                student_probs = torch.log_softmax(self.student(data)/self.temperature, dim=1)
                
                loss = self.kl_div(student_probs, teacher_probs)
                loss.backward()
                optimizer.step()

        return self.student

class CostOptimizer:
    def calculate_savings(self, original_model, distilled_model):
        return {
            'size_reduction': 1 - (distilled_model.size/original_model.size),
            'inference_speedup': original_model.latency/distilled_model.latency,
            'memory_reduction': 1 - (distilled_model.mem_usage/original_model.mem_usage)
        }