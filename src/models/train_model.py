import torch
import torch.nn as nn 

class HealthCareChatbotTrainer:
    def __init__(self, model, train_data_loader, val_data_loader, device, optimizer, scheduler):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_loss = float('inf')

    def train(self, epochs):
        self.model = self.model.to(self.device)

        for epoch in range(epochs):
            self.model.train()
            total_loss, total_accuracy = 0, 0
            
            # iterate over batches
            for _, data in enumerate(self.train_data_loader, 0):
                input_ids = data['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = data['attention_mask'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.long)

                outputs = self.model(input_ids, attention_mask)

                # compute the loss
                loss = nn.CrossEntropyLoss()(outputs, targets)
                total_loss += loss.item()

                # compute the accuracy
                preds = torch.argmax(outputs, dim=1).flatten()
                total_accuracy += (preds == targets).cpu().numpy().mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss = total_loss / len(self.train_data_loader)
            avg_train_acc = total_accuracy / len(self.train_data_loader)

            # validation phase
            self.model.eval()
            total_val_loss, total_val_accuracy = 0, 0
            with torch.no_grad():
                for _, data in enumerate(self.val_data_loader, 0):
                    input_ids = data['input_ids'].to(self.device, dtype=torch.long)
                    attention_mask = data['attention_mask'].to(self.device, dtype=torch.long)
                    targets = data['targets'].to(self.device, dtype=torch.long)

                    outputs = self.model(input_ids, attention_mask)

                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    total_val_loss += loss.item()

                    preds = torch.argmax(outputs, dim=1).flatten()
                    total_val_accuracy += (preds == targets).cpu().numpy().mean()

            avg_val_loss = total_val_loss / len(self.val_data_loader)
            avg_val_acc = total_val_accuracy / len(self.val_data_loader)

            # check if the validation loss has decreased, and save the model if so
            if avg_val_loss < self.best_loss:
                print(f'Validation Loss Decreased({self.best_loss:.6f}--->{avg_val_loss:.6f}) \t Saving The Model')
                self.save_model(f'checkpoint_{epoch}.pt')
                self.best_loss = avg_val_loss

            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.6f}, Train Acc: {avg_train_acc:.6f}, Val Loss: {avg_val_loss:.6f}, Val Acc: {avg_val_acc:.6f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
