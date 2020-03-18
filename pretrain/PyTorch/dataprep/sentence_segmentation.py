import nltk
import os
from tqdm import tqdm
import sys

nltk.download('punkt')

input_file = "/gpfs/gpfs0/groups/mozafari/ruixliu/data/enwiki/wikipedia.txt"
output_file = '/gpfs/gpfs0/groups/mozafari/ruixliu/data/enwiki/wikipedia.segmented.nltk.txt'

doc_seperator = "\n"

with open(input_file) as ifile:
    with open(output_file, "w") as ofile:
        for i, line in tqdm(enumerate(ifile)):
            if line != "\n":
                sent_list = nltk.tokenize.sent_tokenize(line)
                for sent in sent_list:
                    ofile.write(sent + "\n")
                ofile.write(doc_seperator)
