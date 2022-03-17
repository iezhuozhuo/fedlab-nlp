from model.textcnn import TextCNN
from model.bilstm import BiLSTM_TextClassification

from training.utils.register import registry


def create_model(args, model_name, input_size, output_size, embedding_weights):
    logger = registry.get("logger")
    role = registry.get("role")
    logger.info("create %s model. model_name = %s, input_size = %s, label num = %s"
                % (role, model_name, input_size, output_size))

    if model_name == "bilstm_attention":
        model = BiLSTM_TextClassification(input_size, args.hidden_size, output_size, args.num_layers,
                                          args.embedding_dropout, args.lstm_dropout, args.attention_dropout,
                                          args.embedding_length, attention=True, embedding_weights=embedding_weights)
    elif model_name == "bilstm":
        model = BiLSTM_TextClassification(input_size, args.hidden_size, output_size, args.num_layers,
                                          args.embedding_dropout, args.lstm_dropout, args.attention_dropout,
                                          args.embedding_length, embedding_weights=embedding_weights)
    elif model_name == "textcnn":
        model = TextCNN(input_size, args.max_seq_len, output_size, args.embedding_length,
                        dropout=args.cnn_dropout, num_filter=args.num_filter,
                        embedding_dropout=args.embedding_dropout,
                        embedding_weights=embedding_weights)
    else:
        raise Exception("No such model")
    return model