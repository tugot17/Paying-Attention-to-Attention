import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook
import random
import matplotlib.pyplot as plt
from pylab import rcParams
import os
import numpy as np
import matplotlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = os.path.join('human-machine.csv')

df = pd.read_csv(PATH, index_col=0)

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30


class CharDict:
    def __init__(self):
        self.char2index = {}
        self.char2count = {}
        self.index2char = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_chars = 2

    def addSentence(self, sentence):
        for c in sentence:
            self.addChar(c)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1


def prepare_data():
    df = pd.read_csv(PATH)
    human = df["human"].values
    machine = df["machine"].values
    pairs = []
    for i in range(len(human)):
        pairs.append([human[i], machine[i]])

    input_lang = CharDict()
    output_lang = CharDict()
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepare_data()


def indexes_from_sentence(lang, sentence):
    return [lang.char2index[char] for char in sentence]


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1)).to(device)
    # if USE_CUDA: var = var.cuda()
    return var


def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


EPOCHS = 1000
LEARNING_RATE = 0.005
HIDDEN_SIZE = 64
TEACHER_FORCING_RATIO = 0.5



def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target  as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


encoder = EncoderRNN(input_lang.n_chars, HIDDEN_SIZE).to(device)
decoder = AttentionDecoderRNN(HIDDEN_SIZE, output_lang.n_chars, dropout_p=0.1).to(device)


encoder_optimizer = optim.SGD(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LEARNING_RATE)

criterion = nn.NLLLoss()


training_pairs = [variables_from_pair(random.choice(pairs))
                  for i in range(EPOCHS)]


losses = []


for epoch in tqdm_notebook(range(EPOCHS)):
    
    pair = training_pairs[epoch]
    input_tensor = pair[0]
    target_tensor = pair[1]
    

    
    loss = train_step(input_tensor, target_tensor, encoder,
             decoder, encoder_optimizer, decoder_optimizer, criterion)

    if epoch % 1000 == 0:
        print(f"Loss: {loss}")
    
    losses.append(loss)
    


plt.plot(losses)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.grid()
plt.show()


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    droput_value = decoder.dropout_p
    
    with torch.no_grad():
        decoder.dropout_p = 0.0
        
        input_tensor = variable_from_sentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i],
                                                     encoder_hidden)
            encoder_outputs[i] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()
            
        decoder.dropout_p = droput_value

        return decoded_words, decoder_attentions[:di + 1]




def visualize_attention(input_word, output_word, attentions):
    fig, ax = plt.subplots()
    im = ax.imshow(attentions.numpy())

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(input_word)))
    ax.set_yticks(np.arange(len(output_word)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(input_word)
    ax.set_yticklabels(output_word)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    output_word = "".join(output_word).replace("<EOS>", "")

    ax.set_title(f"Input: {input_word}, Prediction: {output_word}")
    fig.tight_layout()

    
    rcParams['figure.figsize'] = 10, 20
    plt.show()
    


input_word = "fr. 3 07 1997"

output_word, attentions = evaluate(
    encoder, decoder, input_word)

visualize_attention(input_word, output_word, attentions)




