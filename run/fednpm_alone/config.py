def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='bilstm', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='sentiment140', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_file', type=str, default="",
                        metavar="DF", help='data h5 file')

    parser.add_argument('--partition_file', type=str, default="",
                        metavar="PF", help='partition h5 file')

    parser.add_argument('--cache_dir', type=str, default="",
                        metavar="DF", help='data h5 file')

    parser.add_argument('--partition_method', type=str, default='uniform', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--hidden_size', type=int, default=300, metavar='H',
                        help='size of hidden layers')

    parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                        help='number of layers in neural network')

    parser.add_argument('--lstm_dropout', type=float, default=0.1, metavar='LD',
                        help="dropout rate for LSTM's output")

    parser.add_argument('--cnn_dropout', type=float, default=0.1, metavar='CLD',
                        help="dropout rate for CNN's output")

    parser.add_argument('--num_filter', type=int, default=50)

    parser.add_argument('--embedding_dropout', type=float, default=0, metavar='ED',
                        help='dropout rate for word embedding')

    parser.add_argument('--attention_dropout', type=float, default=0, metavar='AD',
                        help='dropout rate for attention layer output, only work when BiLSTM_Attention is chosen')

    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_in_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--max_seq_len', type=int, default=512, metavar='MSL',
                        help='maximum sequence length (-1 means the maximum sequence length in the dataset)')

    parser.add_argument('--embedding_file', type=str, default="data/pretrained/glove.840B.300d.txt",
                        metavar="EF", help='word embedding file')

    parser.add_argument('--embedding_name', type=str, default="glove", metavar="EN",
                        help='word embedding name(word2vec, glove)')

    parser.add_argument('--embedding_length', type=int, default=300, metavar="EL", help='dimension of word embedding')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--embedding_lr', type=float, default=2e-5, metavar='ELR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)
    parser.add_argument('--embedding_wd', help='weight decay parameter;', type=float, default=1e-5)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu_server_num', type=int, default=2,
                        help='gpu_server_num')
    #
    parser.add_argument('--gpu_num_per_server', type=int, default=1,
                        help='gpu_num_per_server')

    parser.add_argument('--gpu_num_per_sub_server', type=int, default=1,
                        help='gpu_num_per_sub_server using in scale')

    parser.add_argument("--do_remove_stop_words", type=lambda x: (str(x).lower() == 'true'), default=True,
                        metavar="RSW",
                        help="remove stop words which specify in sapcy")

    parser.add_argument('--do_remove_low_freq_words', type=int, default=0, metavar="RLW",
                        help='remove words in lower frequency')

    parser.add_argument('--do_train', type=lambda x: (str(x).lower() == 'true'), default=False, metavar='N')

    parser.add_argument('--do_test', type=lambda x: (str(x).lower() == 'true'), default=False, metavar='N')

    parser.add_argument('--machine_name', type=str, default='pclt4', metavar='MN',
                        help='machine used in training')

    parser.add_argument('--seed', type=int, default=0, metavar='SD',
                        help='seed used in training')

    parser.add_argument('--grid_search', action='store_true',
                        help="grid_search")

    parser.add_argument('--wandb_time', type=str, default='0')
    parser.add_argument("--wandb_enable", type=lambda x: (str(x).lower() == 'true'), default=False,
                        metavar="WE",
                        help="wandb enable")

    parser.add_argument('--vocabulary_type', type=str, default='part')

    parser.add_argument("--niid", type=lambda x: (str(x).lower() == 'true'), default=False,
                        metavar="Niid",
                        help="niid enable")

    parser.add_argument("--ip", type=str, help="fednpm options.")
    parser.add_argument("--port", type=str, default="", help="fednpm options.")
    parser.add_argument("--world_size", type=int, default=-1, help="fednpm options.")
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--alpha", type=str, default="1.0")
    parser.add_argument("--ethernet", type=str, default=None)
    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')
    parser.add_argument('--ci', type=int, default=None,
                        help='CI')

    args = parser.parse_args()
    return args