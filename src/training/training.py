import os
import zipfile

from utils.hparams import prepare_hparams
from iterators.mind_iterator import MINDIterator
from model.nrms_model import NRMSModel

yaml_file = "/kaggle/input/mind-dataset/nrms.yaml"
wordEmb_file="/kaggle/input/mind-dataset/embedding.npy"
wordDict_file="/kaggle/input/mind-dataset/word_dict.pkl"
userDict_file="/kaggle/input/mind-dataset/uid2index.pkl"
catDict_file="/kaggle/input/mind-dataset/cat_dict.pkl"
subcatDict_file="/kaggle/input/mind-dataset/subcat_dict.pkl"

data_path = "/kaggle/input/mindlarge/"
data_path2 = "/kaggle/input/mindfull/"
train_news_file = os.path.join(data_path2, r'news_full.tsv')
train_behaviors_file = os.path.join(data_path2, r'behaviors_full.tsv')
valid_news_file = os.path.join(data_path, r'news_valid.tsv')
valid_behaviors_file = os.path.join(data_path, r'behaviors_valid.tsv')
test_news_file = os.path.join(data_path, r'news_test.tsv')
test_behaviors_file = os.path.join(data_path, r'behaviors_test.tsv')

epochs = 3
seed = 42
learning_rate = 0.001
batch_size = 1024

hparams = prepare_hparams(yaml_file, 
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file, 
                          userDict_file=userDict_file,
                          catDict_file=catDict_file,
                          subcatDict_file=subcatDict_file,
                          learning_rate=learning_rate,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)

def init_behaviors2(iterator, behaviors_file):
        """init behavior logs given behaviors file.

        Args:
        behaviors_file: path of behaviors file
        """
        iterator.histories = []
        iterator.imprs = []
        iterator.labels = []
        iterator.impr_indexes = []
        iterator.uindexes = []

        with tf.io.gfile.GFile(behaviors_file, "r") as rd:
            impr_index = 0
            for line in rd:
                uid, time, history, impr = line.strip("\n").split(iterator.col_spliter)[-4:]

                history = [iterator.nid2index[i] for i in history.split()]
                history = [0] * (iterator.his_size - len(history)) + history[
                    : iterator.his_size
                ]

                impr_news = [iterator.nid2index[i.split("-")[0]] for i in impr.split()]
                label = [0 for i in impr.split()]
                uindex = iterator.uid2index[uid] if uid in iterator.uid2index else 0

                iterator.histories.append(history)
                iterator.imprs.append(impr_news)
                iterator.labels.append(label)
                iterator.impr_indexes.append(impr_index)
                iterator.uindexes.append(uindex)
                impr_index += 1

def load_impression_from_file2(iterator, behaivors_file):
        """Read and parse impression data from behaivors file.

        Args:
            behaivors_file (str): A file contains several informations of behaviros.

        Yields:
            object: An iterator that yields parsed impression data, in the format of dict.
        """

        init_behaviors2(iterator, behaivors_file)

        indexes = np.arange(len(iterator.labels))

        for index in indexes:
            impr_label = np.array(iterator.labels[index], dtype="int32")
            impr_news = np.array(iterator.imprs[index], dtype="int32")

            yield (
                iterator.impr_indexes[index],
                impr_news,
                iterator.uindexes[index],
                impr_label,
            )

def load_user_from_file2(iterator, news_file, behavior_file):
        """Read and parse user data from news file and behavior file.

        Args:
            news_file (str): A file contains several informations of news.
            beahaviros_file (str): A file contains information of user impressions.

        Yields:
            object: An iterator that yields parsed user feature, in the format of dict.
        """

        iterator.init_news(news_file)

        init_behaviors2(iterator, behavior_file)

        user_indexes = []
        impr_indexes = []
        click_title_indexes = []
        click_abstract_indexes = []
        click_cat_indexes = []
        click_subcat_indexes = []
        cnt = 0

        for index in range(len(iterator.impr_indexes)):
            click_title_indexes.append(iterator.news_title_index[iterator.histories[index]])
            click_abstract_indexes.append(iterator.news_abstract_index[iterator.histories[index]])
            click_cat_indexes.append(iterator.news_cat_index[iterator.histories[index]])
            click_subcat_indexes.append(iterator.news_subcat_index[iterator.histories[index]])
            user_indexes.append(iterator.uindexes[index])
            impr_indexes.append(iterator.impr_indexes[index])

            cnt += 1
            if cnt >= iterator.batch_size:
                yield iterator._convert_user_data(
                    user_indexes,
                    impr_indexes,
                    click_title_indexes,
                    click_abstract_indexes,
                    click_cat_indexes,
                    click_subcat_indexes,
                )
                user_indexes = []
                impr_indexes = []
                click_title_indexes = []
                click_abstract_indexes = []
                click_cat_indexes = []
                click_subcat_indexes = []
                cnt = 0

        if cnt > 0:
            yield iterator._convert_user_data(
                user_indexes,
                impr_indexes,
                click_title_indexes,
                click_abstract_indexes,
                click_cat_indexes,
                click_subcat_indexes
            )

def load_news_from_file2(iterator, news_file):
        print(news_file)
        """Read and parse user data from news file.

        Args:
            news_file (str): A file contains several informations of news.

        Yields:
            object: An iterator that yields parsed news feature, in the format of dict.
        """
        #if not hasattr(iterator, "news_title_index"):
        iterator.init_news(news_file)

        news_indexes = []
        candidate_title_indexes = []
        candidate_abstract_indexes = []
        candidate_cat_indexes = []
        candidate_subcat_indexes = []
        cnt = 0

        for index in range(len(iterator.news_title_index)):
            news_indexes.append(index)
            candidate_title_indexes.append(iterator.news_title_index[index])
            candidate_abstract_indexes.append(iterator.news_abstract_index[index])
            candidate_cat_indexes.append(iterator.news_cat_index[index])
            candidate_subcat_indexes.append(iterator.news_subcat_index[index])
                
            cnt += 1
            if cnt >= iterator.batch_size:
                yield iterator._convert_news_data(
                    news_indexes,
                    candidate_title_indexes,
                    candidate_abstract_indexes,
                    candidate_cat_indexes,
                    candidate_subcat_indexes,
                )
                news_indexes = []
                candidate_title_indexes = []
                candidate_abstract_indexes = []
                candidate_cat_indexes = []
                candidate_subcat_indexes = []
                cnt = 0

        if cnt > 0:
            yield iterator._convert_news_data(
                news_indexes,
                candidate_title_indexes,
                candidate_abstract_indexes,
                candidate_cat_indexes,
                candidate_subcat_indexes
            )

def run_news2(model, news_filename):
        #print(news_filename)
        if not hasattr(model, "newsencoder"):
            raise ValueError("model must have attribute newsencoder")

        news_indexes = []
        news_vecs = []
        for batch_data_input in tqdm(
            load_news_from_file2(model.test_iterator, news_filename)
        ):
            news_index, news_vec = model.news(batch_data_input)
            news_indexes.extend(np.reshape(news_index, -1))
            news_vecs.extend(news_vec)

        return dict(zip(news_indexes, news_vecs))

def run_user2(model, news_filename, behaviors_file):
        if not hasattr(model, "userencoder"):
            raise ValueError("model must have attribute userencoder")

        user_indexes = []
        user_vecs = []
        for batch_data_input in tqdm(
            load_user_from_file2(model.test_iterator, news_filename, behaviors_file)
        ):
            user_index, user_vec = model.user(batch_data_input)
            user_indexes.extend(np.reshape(user_index, -1))
            user_vecs.extend(user_vec)

        return dict(zip(user_indexes, user_vecs))

def run_fast_eval2(model, news_filename, behaviors_file):
        news_vecs = run_news2(model, news_filename)
        #print(news_filename)
        user_vecs = run_user2(model, news_filename, behaviors_file)

        model.news_vecs = news_vecs
        model.user_vecs = user_vecs

        group_impr_indexes = []
        group_labels = []
        group_preds = []

        for (
            impr_index,
            news_index,
            user_index,
            label,
        ) in tqdm(load_impression_from_file2(model.test_iterator, behaviors_file)):
            pred = np.dot(
                np.stack([news_vecs[i] for i in news_index], axis=0),
                user_vecs[impr_index],
            )
            group_impr_indexes.append(impr_index)
            group_labels.append(label)
            group_preds.append(pred)

        return group_impr_indexes, group_labels, group_preds

if __name__ ==  "__main__":
    # detect and init the TPU
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    # instantiate a distribution strategy
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

    iterator = MINDIterator
    with tpu_strategy.scope():
        model = NRMSModel(hparams, iterator)
    model.fit(train_news_file, train_behaviors_file, valid_news_file, valid_behaviors_file)\

    #Submit
    group_impr_indexes, group_labels, group_preds = run_fast_eval2(model, test_news_file, test_behaviors_file)

    output_path = "/kaggle/working/"
    with open(os.path.join(output_path, 'prediction.txt'), 'w') as f:
        for impr_index, preds in tqdm(zip(group_impr_indexes, group_preds)):
            impr_index += 1
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
            f.write(' '.join([str(impr_index), pred_rank])+ '\n')
    
    f = zipfile.ZipFile(os.path.join(output_path, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
    f.write(os.path.join(output_path, 'prediction.txt'), arcname='prediction.txt')
    f.close()