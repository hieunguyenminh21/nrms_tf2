import os

from utils.hparams import prepare_hparams
from iterators.mind_iterator import MINDIterator
from model.nrms_model import NRMSModel

def training():
    data_path = "/content/drive/MyDrive/NRMS/"
    yaml_file = os.path.join(data_path, "nrms.yaml")
    wordEmb_file = os.path.join(data_path, "embedding.npy")
    wordDict_file = os.path.join(data_path, "word_dict.pkl")
    userDict_file = os.path.join(data_path, "uid2index.pkl")

    train_news_file = os.path.join(data_path, 'train', r'news.tsv')
    train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
    valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
    valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')

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
    print(hparams)

    iterator = MINDIterator

    model = NRMSModel(hparams, iterator)

    #%%time
    model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)

    return model
if __name__ ==  "__main__":
    model = training()

