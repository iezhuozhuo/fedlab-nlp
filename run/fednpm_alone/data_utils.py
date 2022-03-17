import os
import torch
from spacy.lang.en import STOP_WORDS

from globalhost import machine_dict

from data_preprocessing.base.utils import *
import data.raw_data_loader.obsolete.AGNews.data_loader
import data.raw_data_loader.obsolete.SST_2.data_loader
import data.raw_data_loader.obsolete.SemEval2010Task8.data_loader
import data.raw_data_loader.obsolete.Sentiment140.data_loader
import data.raw_data_loader.obsolete.news_20.data_loader
import data.raw_data_loader.obsolete.Depression.data_loader
import data.raw_data_loader.obsolete.Finance.data_loader


def load_and_process_dataset(args, dataset_name):
    logger = registry.get("logger")

    file_name = f"{dataset_name}_seq={args.max_seq_len}_" \
                f"niid={args.niid}_num={args.client_num_in_total}_alpha={args.alpha}_" \
                f"embedType={args.vocabulary_type}.h5"
    cache_file = os.path.join(args.cache_dir, file_name)
    # args.cache_file = cache_file

    if os.path.isfile(cache_file):
        logger.info(f"loading cache data from {cache_file}")
        dataset = torch.load(cache_file)
    else:
        logger.info(f"generating cache data to {cache_file}")
        dataset = load_data(args, args.dataset)
        dataset = preprocess_data(args, dataset)
        torch.save(dataset, cache_file)

    # dataset = load_data(args, args.dataset)
    # dataset = preprocess_data(args, dataset)
    return dataset


def load_data(args, dataset_name):
    logger = registry.get("logger")
    logger.warning(f"loading {args.dataset} dataset with niid={args.niid} and alpha={args.alpha}")

    postfix = "pkl"
    if not args.data_file:
        args.data_file = os.path.join(machine_dict[args.machine_name]["data_path"],
            f"{args.dataset}_data.{postfix}")

    if not args.partition_file:
        if args.niid:
            if args.partition_method:
                args.partition_file = os.path.join(machine_dict[args.machine_name]["partition_path"],
                    f"{args.dataset}_partition.{postfix}")
            else:
                args.partition_file = os.path.join(machine_dict[args.machine_name]["partition_path"],
                    f"niid_{args.dataset}_pdata.{postfix}")
                args.partition_method = f"niid_label_clients={args.client_num_in_total}_alpha={args.alpha}"
        else:
            args.partition_file = os.path.join(machine_dict[args.machine_name]["partition_path"],
                f"{args.dataset}_partition.{postfix}")
            args.partition_method = "uniform"

    clients_num = args.client_num_in_total
    client_data_loaders = []

    if dataset_name == "20news":
        server_data_loader = data.raw_data_loader.obsolete.news_20.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
            partition_method=args.partition_method, tokenize=True,
            clients_num=clients_num)
        for client_index in range(args.client_num_in_total):
            client_data_loader = data.raw_data_loader.obsolete.news_20.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                partition_method=args.partition_method, tokenize=True,
                client_idx=client_index,
                clients_num=clients_num)
            client_data_loaders.append(client_data_loader)

    elif dataset_name == "agnews":
        server_data_loader = data.raw_data_loader.obsolete.AGNews.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
            partition_method=args.partition_method, tokenize=True,
            clients_num=clients_num)
        for client_index in range(args.client_num_in_total):
            client_data_loader = data.raw_data_loader.obsolete.AGNews.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                partition_method=args.partition_method, tokenize=True,
                client_idx=client_index,
                clients_num=clients_num)
            client_data_loaders.append(client_data_loader)

    elif dataset_name == "sst_2":
        server_data_loader = data.raw_data_loader.obsolete.SST_2.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
            partition_method=args.partition_method, tokenize=True,
            clients_num=clients_num)

        for client_index in range(args.client_num_in_total):
            client_data_loader = data.raw_data_loader.obsolete.SST_2.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                partition_method=args.partition_method, tokenize=True,
                client_idx=client_index,
                clients_num=clients_num)
            client_data_loaders.append(client_data_loader)

    elif dataset_name == "finance":
        server_data_loader = data.raw_data_loader.obsolete.Finance.data_loader. \
            ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
            partition_method=args.partition_method, tokenize=True,
            clients_num=clients_num)
        for client_index in range(args.client_num_in_total):
            client_data_loader = data.raw_data_loader.obsolete.Finance.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                partition_method=args.partition_method, tokenize=True,
                client_idx=client_index,
                clients_num=clients_num)
            client_data_loaders.append(client_data_loader)

    elif dataset_name == "depression":
        for client_index in range(args.client_num_in_total):
            client_data_loader = data.raw_data_loader.obsolete.Depression.data_loader. \
                ClientDataLoader(os.path.abspath(args.data_file), os.path.abspath(args.partition_file),
                partition_method=args.partition_method, tokenize=True,
                client_idx=client_index,
                clients_num=clients_num)
            client_data_loaders.append(client_data_loader)
    else:
        raise Exception("No such dataset")

    attributes = server_data_loader.get_attributes()
    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    dev_data_local_dict = dict()
    for idx in range(attributes["n_clients"]):
        train_data_local_num_dict[idx] = client_data_loaders[idx].get_train_data_num()
        train_data_local_dict[idx] = client_data_loaders[idx].get_train_batch_data(args.batch_size)
        test_data_local_dict[idx] = client_data_loaders[idx].get_test_batch_data(args.batch_size*4)
        dev_data_local_dict[idx] = client_data_loaders[idx].get_dev_batch_data(args.batch_size)

    train_data_global = server_data_loader.get_train_batch_data(args.batch_size)
    test_data_global = server_data_loader.get_test_batch_data(args.batch_size*4)
    dev_data_global = server_data_loader.get_dev_batch_data(args.batch_size)

    dataset = [server_data_loader.get_train_data_num(),
               server_data_loader.get_test_data_num(),
               int(server_data_loader.get_dev_data_num()/100),
               train_data_global, test_data_global, dev_data_global,
               train_data_local_num_dict, train_data_local_dict,
               dev_data_local_dict, test_data_local_dict, attributes]
    return dataset


def preprocess_data(args, dataset):
    logger = registry.get("logger")
    logger.info("preproccess data")
    [train_data_num, test_data_num, dev_data_num,
     train_data_global, test_data_global, dev_data_global,
     train_data_local_num_dict, train_data_local_dict, dev_data_local_dict, test_data_local_dict,
     attributes] = dataset

    target_vocab = attributes["label_vocab"]
    # remove low frequency words and stop words
    # build frequency vocabulary based on tokenized data
    x = []
    for i, batch_data in enumerate(train_data_global):
        x.extend(batch_data["X"])
    for i, batch_data in enumerate(test_data_global):
        x.extend(batch_data["X"])
    for i, batch_data in enumerate(dev_data_global):
        x.extend(batch_data["X"])
    freq_vocab = build_freq_vocab(x)
    logger.info(f"frequency vocab size {len(freq_vocab)}")
    registry.register("all_vocb_size", len(freq_vocab))

    def __remove_words(dataset, word_set):
        if isinstance(dataset, dict):
            for client_index in dataset.keys():
                for i, batch_data in enumerate(dataset[client_index]):
                    dataset[client_index][i]["X"] = remove_words(batch_data["X"], word_set)
        else:
            for i, batch_data in enumerate(dataset):
                dataset[i]["X"] = remove_words(batch_data["X"], word_set)

    if args.do_remove_low_freq_words > 0:
        logger.warning("remove low frequency words")
        # build low frequency words set
        low_freq_words = set()
        for token, freq in freq_vocab.items():
            if freq <= args.do_remove_low_freq_words:
                low_freq_words.add(token)
        __remove_words(train_data_global, low_freq_words)
        __remove_words(test_data_global, low_freq_words)
        __remove_words(dev_data_global, low_freq_words)
        __remove_words(train_data_local_dict, low_freq_words)
        __remove_words(test_data_local_dict, low_freq_words)
        __remove_words(dev_data_local_dict, low_freq_words)

    if args.do_remove_stop_words:
        logger.warning("remove stop words")
        __remove_words(train_data_global, STOP_WORDS)
        __remove_words(test_data_global, STOP_WORDS)
        __remove_words(dev_data_global, STOP_WORDS)
        __remove_words(train_data_local_dict, STOP_WORDS)
        __remove_words(test_data_local_dict, STOP_WORDS)
        __remove_words(dev_data_local_dict, STOP_WORDS)

    x.clear()
    for i, batch_data in enumerate(train_data_global):
        x.extend(batch_data["X"])
    for i, batch_data in enumerate(test_data_global):
        x.extend(batch_data["X"])
    for i, batch_data in enumerate(dev_data_global):
        x.extend(batch_data["X"])
    source_vocab = build_vocab(x)
    logger.info(f"source vocab size {len(source_vocab)}")
    registry.register("training_vocb_size", len(source_vocab) + 2)

    # load pretrained embeddings. Note that we use source vocabulary here to reduce the input size
    embedding_weights = None
    if args.embedding_name:
        args.embedding_file = os.path.join(machine_dict[args.machine_name]["pretrain_path"],
            args.embedding_file)
        if args.embedding_name == "word2vec":
            logger.info(f"load word embedding {args.embedding_name}")
            source_vocab, embedding_weights = load_word2vec_embedding(os.path.abspath(args.embedding_file),
                source_vocab)
        elif args.embedding_name == "glove":
            logger.info(f"load word embedding {args.embedding_name}")
            if args.vocabulary_type == "all":
                source_vocab, embedding_weights = load_glove_embedding(os.path.abspath(args.embedding_file),
                    source_vocab=None,
                    dimension=args.embedding_length)
            else:
                source_vocab, embedding_weights = personal_client_load_glove_embedding(os.path.abspath(args.embedding_file),
                    source_vocab=source_vocab,
                    dimension=args.embedding_length)
        else:
            raise Exception("No such embedding")
        embedding_weights = torch.tensor(embedding_weights, dtype=torch.float)

    if args.max_seq_len == -1:
        lengths = []
        for batch_data in train_data_global:
            lengths.extend([len(single_x) for single_x in batch_data["X"]])
        args.max_seq_len = max(lengths)

    def __padding_batch(data):
        new_data = list()
        for i, batch_data in enumerate(data):
            padding_x, seq_lens = padding_data(batch_data["X"], args.max_seq_len)
            new_data.append({"X": token_to_idx(padding_x, source_vocab),
                             "Y": label_to_idx(batch_data["Y"], target_vocab),
                             "seq_lens": seq_lens})
        return new_data

    train_data_global = __padding_batch(train_data_global)
    test_data_global = __padding_batch(test_data_global)
    dev_data_global = __padding_batch(dev_data_global)

    for client_index in train_data_local_dict.keys():
        train_data_local_dict[client_index] = __padding_batch(train_data_local_dict[client_index])
        test_data_local_dict[client_index] = __padding_batch(test_data_local_dict[client_index])
        dev_data_local_dict[client_index] = __padding_batch(dev_data_local_dict[client_index])

    logger.info(f"size of source vocab: {len(source_vocab)}, size of target label: {len(target_vocab)}")
    dataset = [train_data_num, test_data_num, dev_data_num,
               train_data_global, test_data_global, dev_data_global,
               train_data_local_num_dict, train_data_local_dict,
               dev_data_local_dict, test_data_local_dict,
               source_vocab, target_vocab, embedding_weights]
    return dataset
