dataset_dir = "data/dataset.csv"

# Parameters/PATHs for GPT2 model
save_model_dir = "weights/"
model_name = "gpt2"
label_ids = {"neg": 0, "pos": 1}
num_labels = len(label_ids)
epochs = 4
batch_size = 32
max_length = 60
save_interval = 1