from util import *
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import transformers
from transformers import GPT2Config, GPT2Tokenizer, get_linear_schedule_with_warmup, GPT2ForSequenceClassification
from sklearn.metrics import accuracy_score


class GPT2_classifier():
    """
    """
    def __init__(self,
                 dataset):
        """
        """
        # Set seed and device
        transformers.set_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load GPT2 model
        model_config = GPT2Config.from_pretrained(model_name, 
                                                num_labels=num_labels)
        self.model = GPT2ForSequenceClassification.from_pretrained(model_name, 
                                                            config=model_config)

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Load model on device
        self.model.to(self.device)
        print("Model loaded on ", self.device)
        
        # Set up optimizer
        self.optimizer = AdamW(self.model.parameters(),
                               lr = 2e-5)
        
        # Load dataset as dataloader
        dataset = self.dataset_wrapper(dataset)
        self.dataloader = DataLoader(dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     collate_fn=self.collator)

        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps = 0, 
                                                         num_training_steps = len(self.dataloader) * epochs)
    
    class dataset_wrapper(Dataset):
        """
        Wrapper class for passing dataset to DataLoader. 
        """
        def __init__(self, dataset):
            self.X = dataset["X"]
            self.y = dataset["y"]

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return {"X": self.X[index], 
                    "y" : self.y[index]}
        
    def collator(self, dataset):
        """
        Collator for DataLoader.
        """        
        # Call tokenizer on text
        inputs = self.tokenizer(text=[data["X"] for data in dataset], 
                                return_tensors="pt", 
                                padding=True, 
                                truncation=True,  
                                max_length=max_length)
        
        inputs.update({"labels" : torch.tensor([data["y"] for data in dataset])})
        
        return inputs
    
    def train(self):
        """
        Train model on sequential classification task.
        """        
        self.model.train()
        
        for epoch in tqdm(range(epochs)):
            predictions_labels = []
            true_labels = []
            total_loss = 0
            
            for batch in tqdm(self.dataloader, total=len(self.dataloader)):                
                # Add original labels - use later for evaluation.
                true_labels += batch["labels"].numpy().flatten().tolist()

                # move batch to device
                batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}

                self.model.zero_grad()
                outputs = self.model(**batch)
                loss, logits = outputs[:2]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                logits = logits.detach().cpu().numpy()
                predictions_labels += logits.argmax(axis=-1).flatten().tolist()
            
            avg_epoch_loss = total_loss/len(self.dataloader)    
            train_acc = accuracy_score(true_labels, predictions_labels)
            
            print("Epoch: {:>12}  Avg. Loss: {:>12}  Training Acc.: {:>12}".format(epoch, avg_epoch_loss, train_acc))
            
            if epoch % save_interval == 0:
                torch.save(self.model.state_dict(), save_model_dir)
