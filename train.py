from tokenizer.tokenizer import Vocab, TextEncoder

# read it in to inspect it
with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = Vocab(text)
encoder = TextEncoder(vocab)

encoded_numbers = encoder.encode(text)
decoded_text = encoder.decode(encoded_numbers)
tensor_dataset = encoder.convert_dataset_to_tensor(text)

# Let's now split up the data into train and validation sets
n = int(0.9*len(tensor_dataset)) # first 90% will be train, rest val
train_data = tensor_dataset[:n]
val_data = tensor_dataset[n:]
x=1