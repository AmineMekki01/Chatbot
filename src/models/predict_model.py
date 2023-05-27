import torch

class HealthCareChatbotPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, input_data):
        self.model.eval()
        input_ids = input_data['input_ids'].to(self.device, dtype=torch.long)
        attention_mask = input_data['attention_mask'].to(self.device, dtype=torch.long)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        _, predicted = torch.max(outputs, dim=1)

        return predicted
    
