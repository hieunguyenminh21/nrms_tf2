import yaml
from collections import namedtuple


def load_yaml(filename):
    """Load yaml file
    Args:
        filename
    Returns:
        dict
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:
        raise
    except Exception as e:
        raise IOError(f"load {filename} error!")


def flat_config(config):
    """Flat config load tu yaml file to flat dict 

    Args:
        config (dict): config load tu yaml file
    Returns:
        dict
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def check_type(config):
    int_parameters = [
        "word_size",
        "his_size",
        "title_size",
        "body_size",
        "npratio",
        "word_emb_dim",
        "attention_hidden_dim",
        "epochs",
        "batch_size",
        "show_step",
        "save_epoch",
        "head_num",
        "head_dim",
        "user_num",
        "filter_num",
        "window_size",
        "gru_unit",
        "user_emb_dim",
        "vert_emb_dim",
        "subvert_emb_dim",
    ]
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("Parameters {0} must be int".format(param))

    float_parameters = ["learning_rate", "dropout"]
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("Parameters {0} must be float".format(param))  
        str_parameters = [
        "wordEmb_file",
        "wordDict_file",
        "userDict_file",
        "vertDict_file",
        "subvertDict_file",
        "method",
        "loss",
        "optimizer",
        "cnn_activation",
        "dense_activation" "type",
    ]
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("Parameters {0} must be str".format(param))

    list_parameters = ["layer_sizes", "activation"]
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("Parameters {0} must be list".format(param))

    bool_parameters = ["support_quick_scoring"]
    for param in bool_parameters:
        if param in config and not isinstance(config[param], bool):
            raise TypeError("Parameters {0} must be bool".format(param))


def check_nn_config(f_config):
    """Check neural net config
    Args:
        f_config (dict): file config duoc flat tu yaml file
    Raises:
        ValueError: Neu params bi sai -> Raise error
    """
    if f_config["model_type"] in ["nrms", "NRMS"]:
        required_parameters = [
            "title_size",
            "his_size",
            "wordEmb_file",
            "wordDict_file",
            "userDict_file",
            "npratio",
            "data_format",
            "word_emb_dim",
            "head_num",
            "head_dim",
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout",
        ]
    else:
        required_parameters = []
    
    #check
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("Parameter {0} must be set!".format(param))
    if f_config["model_type"] in ["nrms", "NRMS"]:
        if f_config["data_format"] != "news":
            raise ValueError(
                "Voi NRMS model, dataformat phai la news, do ngu, may lai dua cai {0} vao lam gi".format(f_config["data_format"])
            )
    
    check_type(f_config)


def get_hparams(**kwargs):
    return namedtuple('GenericDict', kwargs.keys())(**kwargs)


def create_hparams(flags):
    """Create model's params
    Args:
        flags (dict): Dict co requirement
    Returns:
        object: namedtuple
    """
    return get_hparams(
        #data
        data_format=flags.get("data_format", None),
        iterator_type=flags.get("iterator_type", None),
        support_quick_scoring=flags.get("support_quick_scoring", False),
        wordEmb_file=flags.get("wordEmb_file", None),
        wordDict_file=flags.get("wordDict_file", None),
        userDict_file=flags.get("userDict_file", None),
        catDict_file=flags.get("catDict_file", None),
        subcatDict_file=flags.get("subcatDict_file", None),
        # models
        title_size=flags.get("title_size", None),
        body_size=flags.get("body_size", None),
        word_emb_dim=flags.get("word_emb_dim", None),
        word_size=flags.get("word_size", None),
        user_num=flags.get("user_num", None),
        cat_num=flags.get("cat_num", None),
        subcat_num=flags.get("subcat_num", None),
        his_size=flags.get("his_size", None),
        npratio=flags.get("npratio"),
        dropout=flags.get("dropout", 0.0),
        attention_hidden_dim=flags.get("attention_hidden_dim", 200),
        # nrms
        head_num=flags.get("head_num", 4),
        head_dim=flags.get("head_dim", 100),
        #train
        learning_rate=flags.get("learning_rate", 0.001),
        loss=flags.get("loss", None),
        optimizer=flags.get("optimizer", "adam"),
        epochs=flags.get("epochs", 10),
        batch_size=flags.get("batch_size", 1),
        # show info
        show_step=flags.get("show_step", 1),
        metrics=flags.get("metrics", None),
    )

def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare hyperparams and make sure it's ok
    Args:
        yaml_file: path to yaml file
    Returns:
        TF Hyperparams object (tf.contrib.training.HParams)
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}
    
    config.update(kwargs)
    check_nn_config(config)
    return create_hparams(config)


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare hyperparams and make sure it's ok
    Args:
        yaml_file: path to yaml file
    Returns:
        TF Hyperparams object (tf.contrib.training.HParams)
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}
    
    config.update(kwargs)
    check_nn_config(config)
    return create_hparams(config)
