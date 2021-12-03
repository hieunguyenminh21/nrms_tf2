import os

from utils.hparams import prepare_hparams
from iterators.mind_iterator import MINDIterator
from model.nrms_model import NRMSModel

data_path = "/home/hieunm/VCCorp/paper/nrms_tf2/data/"
yaml_file = os.path.join(data_path, "nrms.yaml")
wordEmb_file = os.path.join(data_path, "embedding.npy")
wordDict_file = os.path.join(data_path, "word_dict.pkl")
userDict_file = os.path.join(data_path, "uid2index.pkl")

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
test_behaviors_file = os.path.join(data_path, 'test', r'behaviors.tsv')
test_news_file = os.path.join(data_path, 'test', r'news.tsv')

epochs = 5
seed = 42
batch_size = 32

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)

if __name__ ==  "__main__":
    iterator = MINDIterator
    model = NRMSModel(hparams, iterator)
    model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)
    