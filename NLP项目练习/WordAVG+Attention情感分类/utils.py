import torch


def tokenizer(sentence):
    return sentence.split()


def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time-(elapsed_mins*60))
    return elapsed_mins, elapsed_secs


def binary_accuracy(score, y):
    round_score = torch.round(torch.sigmoid(score))
    correct = (round_score == y).float()
    acc = correct.sum()/len(correct)
    return acc
