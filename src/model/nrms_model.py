# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import keras
from tensorflow.keras import layers

from model.base_model import BaseModel
from layers.layers import AttLayer2, SelfAttention

__all__ = ["NRMSModel"]


class NRMSModel(BaseModel):
    """NRMS model(Neural News Recommendation with Multi-Head Self-Attention)

    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang,and Xing Xie, "Neural News
    Recommendation with Multi-Head Self-Attention" in Proceedings of the 2019 Conference
    on Empirical Methods in Natural Language Processing and the 9th International Joint Conference
    on Natural Language Processing (EMNLP-IJCNLP)

    Attributes:
        word2vec_embedding (numpy.ndarray): Pretrained word embedding matrix.
        hparam (object): Global hyper-parameters.
    """

    def __init__(
        self,
        hparams,
        iterator_creator,
        seed=None,
    ):
        """Initialization steps for NRMS.
        Compared with the BaseModel, NRMS need word embedding.
        After creating word embedding matrix, BaseModel's __init__ method will be called.

        Args:
            hparams (object): Global hyper-parameters. Some key setttings such as head_num and head_dim are there.
            iterator_creator_train (object): NRMS data loader class for train data.
            iterator_creator_test (object): NRMS data loader class for test and validation data
        """
        self.word2vec_embedding = self._init_embedding(hparams.wordEmb_file)

        super().__init__(
            hparams,
            iterator_creator,
            seed=seed,
        )

    def _get_input_label_from_iter(self, batch_data):
        """get input and labels for trainning from iterator

        Args:
            batch data: input batch data from iterator

        Returns:
            list: input feature fed into model (clicked_title_batch & candidate_title_batch)
            numpy.ndarray: labels
        """
        input_feat = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_abstract_batch"],
            batch_data["clicked_cat_batch"],
            batch_data["clicked_subcat_batch"],
            batch_data["candidate_title_batch"],
            batch_data["candidate_abstract_batch"],
            batch_data["candidate_cat_batch"],
            batch_data["candidate_subcat_batch"]
        ]
        input_label = batch_data["labels"]
        return input_feat, input_label

    def _get_user_feature_from_iter(self, batch_data):
        """get input of user encoder
        Args:
            batch_data: input batch data from user iterator

        Returns:
            numpy.ndarray: input user feature (clicked title batch)
        """
        input_feature = [
            batch_data["clicked_title_batch"],
            batch_data["clicked_abstract_batch"],
            batch_data["clicked_cat_batch"],
            batch_data["clicked_subcat_batch"]
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _get_news_feature_from_iter(self, batch_data):
        """get input of news encoder
        Args:
            batch_data: input batch data from news iterator

        Returns:
            numpy.ndarray: input news feature (candidate title batch)
        """
        input_feature = [
            batch_data["candidate_title_batch"],
            batch_data["candidate_abstract_batch"],
            batch_data["candidate_cat_batch"],
            batch_data["candidate_subcat_batch"]
        ]
        input_feature = np.concatenate(input_feature, axis=-1)
        return input_feature

    def _build_graph(self):
        """Build NRMS model and scorer.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams
        model, scorer = self._build_nrms()
        return model, scorer

    def _build_userencoder(self, newsencoder):
        """The main function to create user encoder of NRMS.

        Args:
            newsencoder (object): the news encoder of NRMS.

        Return:
            object: the user encoder of NRMS.
        """
        hparams = self.hparams
        his_input_title_abstract_cats = keras.Input(
            shape=(hparams.his_size, hparams.title_size + hparams.body_size + 2),
            dtype="int32"
        )
        
        click_new_presents = layers.TimeDistributed(newsencoder)(his_input_title_abstract_cats)

        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)(
            [click_new_presents] * 3
        )
        
        user_present = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)

        model = keras.Model(his_input_title_abstract_cats, user_present, name="user_encoder")
        print("\n", model.summary())
        return model

    def _build_newsencoder(self, embedding_layer):
        """The main function to create news encoder of NRMS.

        Args:
            embedding_layer (object): a word embedding layer.

        Return:
            object: the news encoder of NRMS.
        """
        hparams = self.hparams

        input_title_abstract_cats = keras.Input(
            shape=(hparams.title_size + hparams.body_size + 2, ), dtype="int32"
        )
        
        sequences_input_title = layers.Lambda(lambda x: x[:, : hparams.title_size])(
            input_title_abstract_cats
        )

        sequences_input_abstract = layers.Lambda(
            lambda x: x[:, hparams.title_size : hparams.title_size + hparams.body_size]
        )(input_title_abstract_cats)
        
        input_cat = layers.Lambda(
            lambda x: x[:, hparams.title_size + hparams.body_size : hparams.title_size + hparams.body_size + 1]
        )(input_title_abstract_cats)
        
        input_subcat = layers.Lambda(
            lambda x: x[:, hparams.title_size + hparams.body_size + 1:]
        )(input_title_abstract_cats)
        
                
        title_rep = self._build_titleencoder(embedding_layer)(sequences_input_title)
        abstract_rep = self._build_abstractencoder(embedding_layer)(sequences_input_abstract)
        cat_rep = self._build_catencoder()(input_cat)
        subcat_rep = self._build_subcatencoder()(input_subcat)
        
        concate_rep = layers.Concatenate(axis=-2)(
            [title_rep, abstract_rep, cat_rep, subcat_rep]
        )

        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([concate_rep, concate_rep, concate_rep])
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        model = keras.Model(input_title_abstract_cats, pred_title, name="news_encoder")
        print("\n", model.summary())
        return model
    
    def _build_titleencoder(self, embedding_layer):

        hparams = self.hparams
        sequences_input_title = keras.Input(shape=(hparams.title_size,), dtype="int32")

        embedded_sequences_title = embedding_layer(sequences_input_title)
  
        y = layers.Dropout(hparams.dropout)(embedded_sequences_title)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)
        pred_title = AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_title = layers.Reshape((1, 400))(pred_title)

        model = keras.Model(sequences_input_title, pred_title, name="title_encoder")
        print("\n", model.summary())
        return model
    
    def _build_abstractencoder(self, embedding_layer):
        hparams = self.hparams

        sequences_input_abstract = keras.Input(shape=(hparams.body_size,), dtype="int32")

        embedded_sequences_abstract = embedding_layer(sequences_input_abstract)
  
        y = layers.Dropout(hparams.dropout)(embedded_sequences_abstract)
        y = SelfAttention(hparams.head_num, hparams.head_dim, seed=self.seed)([y, y, y])
        y = layers.Dropout(hparams.dropout)(y)
        pred_abstract= AttLayer2(hparams.attention_hidden_dim, seed=self.seed)(y)
        pred_abstract = layers.Reshape((1, 400))(pred_abstract)

        model = keras.Model(sequences_input_abstract, pred_abstract, name="abstract_encoder")
        print("\n", model.summary())
        return model

    def _build_catencoder(self):
        hparams = self.hparams
        input_cat = keras.Input(shape = (1, ), dtype = "int32")
        
        cat_embedding = layers.Embedding(
            hparams.cat_num, 400, trainable=True
        )
        
        cat_emb = cat_embedding(input_cat)
        pred_cat = layers.Reshape((1, 400))(cat_emb)
        
        model = keras.Model(input_cat, pred_cat, name = "cat_encoder")
        return model
        
    def _build_subcatencoder(self):
        hparams = self.hparams
        input_subcat = keras.Input(shape = (1, ), dtype = "int32")
        
        subcat_embedding = layers.Embedding(
            hparams.subcat_num, 400, trainable = True
        )
        
        subcat_emb = subcat_embedding(input_subcat)
        pred_subcat = layers.Reshape((1, 400))(subcat_emb)
        
        model = keras.Model(input_subcat, pred_subcat, name = "subcat_encoder")
        return model
        
    def _build_nrms(self):
        """The main function to create NRMS's logic. The core of NRMS
        is a user encoder and a news encoder.

        Returns:
            object: a model used to train.
            object: a model used to evaluate and inference.
        """
        hparams = self.hparams

        his_input_title = keras.Input(
            shape=(hparams.his_size, hparams.title_size), dtype="int32"
        )

        his_input_abstract = keras.Input(
            shape=(hparams.his_size, hparams.body_size), dtype="int32"
        )
        
        his_input_cat = keras.Input(
            shape=(hparams.his_size, 1), dtype="int32"
        )
        
        his_input_subcat = keras.Input(
            shape=(hparams.his_size, 1), dtype="int32"
        )
        
        pred_input_title = keras.Input(
            shape=(hparams.npratio + 1, hparams.title_size), dtype="int32"
        )

        pred_input_abstract = keras.Input(
            shape=(hparams.npratio + 1, hparams.body_size), dtype="int32"
        )
        
        pred_input_cat = keras.Input(
            shape=(hparams.npratio + 1, 1), dtype="int32"
        )
        
        pred_input_subcat = keras.Input(
            shape=(hparams.npratio + 1, 1), dtype="int32"
        )
    
        pred_input_title_one = keras.Input(
            shape=(
                1,
                hparams.title_size,
            ),
            dtype="int32",
        )

        pred_input_abstract_one = keras.Input(
            shape=(
                1,
                hparams.body_size,
            ),
            dtype="int32",
        )
        
        pred_input_cat_one = keras.Input(shape=(1, 1), dtype="int32")
        pred_input_subcat_one = keras.Input(shape=(1, 1), dtype="int32")
        
        his_title_abstract_cats = layers.Concatenate(axis=-1)(
            [his_input_title, his_input_abstract, his_input_cat, his_input_subcat]
        )

        pred_title_abstract_cats = layers.Concatenate(axis=-1)(
            [pred_input_title, pred_input_abstract, pred_input_cat, pred_input_subcat]
        )

        pred_title_abstract_cats_one = layers.Concatenate(axis=-1)(
            [
                pred_input_title_one,
                pred_input_abstract_one,
                pred_input_cat_one,
                pred_input_subcat_one
            ]
        )

        pred_title_abstract_cats_one = layers.Reshape((-1,))(
            pred_title_abstract_cats_one
        )

        embedding_layer = layers.Embedding(
            self.word2vec_embedding.shape[0],
            hparams.word_emb_dim,
            weights=[self.word2vec_embedding],
            trainable=True,
        )

        self.newsencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(self.newsencoder)

        user_present = self.userencoder(his_title_abstract_cats)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_title_abstract_cats)
        news_present_one = self.newsencoder(pred_title_abstract_cats_one)

        preds = layers.Dot(axes=-1)([news_present, user_present])
        preds = layers.Activation(activation="softmax")(preds)

        pred_one = layers.Dot(axes=-1)([news_present_one, user_present])
        pred_one = layers.Activation(activation="sigmoid")(pred_one)

        model = keras.Model([
            his_input_title,
            his_input_abstract,
            his_input_cat,
            his_input_subcat,
            pred_input_title,
            pred_input_abstract,
            pred_input_cat,
            pred_input_subcat
        ], preds)
        #model = keras.Model([his_input_title, pred_input_title], preds)
        scorer = keras.Model([
            his_input_title,
            his_input_abstract,
            his_input_cat,
            his_input_subcat,
            pred_input_title_one,
            pred_input_abstract_one,
            pred_input_cat_one,
            pred_input_subcat_one
        ], pred_one)
        #scorer = keras.Model([his_input_title, pred_input_title_one], pred_one)
        print("\n", model.summary())
        return model, scorer