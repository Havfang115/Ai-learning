import torch

with open("C:/Users/48697/Desktop/ai/Ai-learning/Practice/Text/1342-0.txt", encoding='utf-8') as f:
    text = f.read()

lines = text.split('\n')
line = lines[200]

letter_tensor = torch.zeros(len(line), 128) # 128是由于ASCII的限制

for i, letter in enumerate(line.lower().strip()):
    # 文本里含有双引号，不是有效的ASCII，因此在此处将其屏蔽
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_tensor[i][letter_index] = 1

def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list
words_in_line = clean_words(line)

word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}

word_tensor = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_tensor[i][word_index] = 1
    print('{:2} {:4} {}'.format(i, word_index, word))
