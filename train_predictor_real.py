import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MuSigmaPredictor(nn.Module):
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", freeze_base=True):
        super().__init__()
        self.qwen = AutoModel.from_pretrained(model_name)
        print(self.qwen)
        # This is the hidden size of the Qwen model's output embeddings
        hidden_size = self.qwen.config.hidden_size  # For Qwen-1.5B, this is typically 2048
        
        # Freeze base model parameters if specified
        if freeze_base:
            # Freeze all base model params except the last 2 layers
            for name, param in self.qwen.named_parameters():
                # Only unfreeze the last 2 transformer layers (26 and 27)
                if "layers.26" not in name and "layers.27" not in name:
                    param.requires_grad = False
        
        # Improved prediction head for mu and sigma
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Separate predictor for mu
        self.mu_predictor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # 1 output: mu
        )
        
        # Separate predictor for sigma
        self.sigma_predictor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # 1 output: sigma
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use attention pooling to get a weighted sum of token representations
        attention_weights = self.attention_pooling(hidden_states)
        pooled_output = torch.sum(attention_weights * hidden_states, dim=1)
        
        # Get predictions from separate networks
        mu = self.mu_predictor(pooled_output)
        sigma = self.sigma_predictor(pooled_output)
        
        # Concatenate predictions
        predictions = torch.cat([mu, sigma], dim=1)
        return predictions

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=768):  # Increased max_length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def train_model(model, train_loader, val_loader, device, num_epochs=40, patience=3):
    # Parameters
    lr = 8e-6
    weight_decay = 0.01  # L2 regularization
    grad_clip = 1.0  # Gradient clipping
    
    # Prepare optimizer with different learning rates for different parts
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {
    #         'params': [p for n, p in model.named_parameters() 
    #                   if not any(nd in n for nd in no_decay) and p.requires_grad],
    #         'weight_decay': weight_decay
    #     },
    #     {
    #         'params': [p for n, p in model.named_parameters() 
    #                   if any(nd in n for nd in no_decay) and p.requires_grad],
    #         'weight_decay': 0.0
    #     }
    # ]
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss function
    # optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    # Training tracking
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints_direct_qwen_real', exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # Add L1 regularization for better feature selection
            # l1_lambda = 0.001
            # l1_norm = sum(p.abs().sum() for p in model.predictor.parameters())
            # loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Save checkpoint after every epoch
        checkpoint_path = os.path.join('checkpoints_direct_qwen_real', f'epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Check for improvement and save best model separately
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best val loss: {best_val_loss:.4f}")
            torch.save(model.state_dict(), 'best_predictor_model_direct_qwen_real.pt')
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves_direct_qwen_real.png')
    plt.close()
    
    return train_losses, val_losses

def main():
    # Read the parameters from separate files for train and validation sets
    with open('train_mu_sigma.json', 'r') as f:
        train_mu_sigma_params = json.load(f)
    
    with open('val_mu_sigma.json', 'r') as f:
        val_mu_sigma_params = json.load(f)
    
    # Load the train and validation datasets
    with open('train_data.json', 'r') as f:
        train_dataset = json.load(f)
    
    with open('val_data.json', 'r') as f:
        val_dataset = json.load(f)
    
    # Initialize tokenizer and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model with most layers frozen
    model = MuSigmaPredictor(model_name, freeze_base=True).to(device)
    
    # Prepare data
    train_texts = [item['problem'] for item in train_dataset]
    train_labels = [[params[0], params[1]] for params in train_mu_sigma_params]
    
    val_texts = [item['problem'] for item in val_dataset]
    val_labels = [[params[0], params[1]] for params in val_mu_sigma_params]
    
    # Print dataset sizes and first few examples
    print(f"Train dataset size: {len(train_texts)} samples")
    print(f"Validation dataset size: {len(val_texts)} samples")
    print(f"Example train text: {train_texts[0][:100]}...")
    print(f"Train label: {train_labels[0]}")
    print(f"Example val text: {val_texts[0][:100]}...")
    print(f"Val label: {val_labels[0]}")
    
    print(f"Train texts length: {len(train_texts)}, Train labels length: {len(train_labels)}")
    print(f"Val texts length: {len(val_texts)}, Val labels length: {len(val_labels)}")
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    
    # Create dataloaders with smaller batch size
    batch_size = 16  # Reduced batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Train the model
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Load the best model and evaluate
    model.load_state_dict(torch.load('best_predictor_model_direct_qwen_real.pt'))
    model.eval()
    
    # Calculate final validation metrics
    final_loss = 0
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = nn.MSELoss()(outputs, labels)
            final_loss += loss.item()
            
            # Store predictions and ground truth
            pred_cpu = outputs.cpu().numpy()
            label_cpu = labels.cpu().numpy()
            
            predictions.extend(pred_cpu.tolist())
            ground_truth.extend(label_cpu.tolist())
    
    avg_final_loss = final_loss / len(val_loader)
    print(f"Final validation loss: {avg_final_loss:.4f}")
    
    # Calculate MSE
    predictions_np = np.array(predictions)
    ground_truth_np = np.array(ground_truth)
    
    mu_mse = np.mean((predictions_np[:, 0] - ground_truth_np[:, 0])**2)
    sigma_mse = np.mean((predictions_np[:, 1] - ground_truth_np[:, 1])**2)
    total_mse = np.mean(np.sum((predictions_np - ground_truth_np)**2, axis=1))
    
    print(f"MSE - μ: {mu_mse:.4f}, σ: {sigma_mse:.4f}, Total: {total_mse:.4f}")
    
    # Calculate mean absolute error
    mu_mae = np.mean(np.abs(predictions_np[:, 0] - ground_truth_np[:, 0]))
    sigma_mae = np.mean(np.abs(predictions_np[:, 1] - ground_truth_np[:, 1]))
    total_mae = np.mean(np.sum(np.abs(predictions_np - ground_truth_np), axis=1))
    
    print(f"MAE - μ: {mu_mae:.4f}, σ: {sigma_mae:.4f}, Total: {total_mae:.4f}")
    
    # Save predictions for analysis
    results = {
        'predictions': predictions,
        'ground_truth': ground_truth
    }
    
    with open('validation_results_direct_qwen_real.json', 'w') as f:
        json.dump(results, f)
    
    print("Model training completed and evaluation results saved.")

if __name__ == "__main__":
    main()

