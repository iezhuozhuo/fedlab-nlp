import os

from globalhost import machine_dict
from model.transformer.model_args import ClassificationArgs
from data_manager.data_attributes import tc_data_attributes

from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertForTokenClassification,
    DistilBertForSequenceClassification,
    DistilBertForQuestionAnswering,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
)

MODEL_CLASSES = {
    "classification": {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
        # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
        "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    },
    "seq_tagging": {
        "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
    },
    "span_extraction": {
        "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    },
    "seq2seq": {
        "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
    }
}


def add_model_args(args):
    model_args = ClassificationArgs()
    model_args.model_name = os.path.join(machine_dict[args.machine_name]["pretrained_model_path"],
                                         args.model_name)
    cached_dir_name = args.model_name + f"-world_size={args.world_size}"
    model_args.cache_dir = os.path.join(machine_dict[args.machine_name]["cache_dir"],
                                        cached_dir_name)
    os.makedirs(model_args.cache_dir, exist_ok=True)

    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    num_labels = tc_data_attributes[args.dataset]
    model_args.num_labels = num_labels
    model_args.update_from_dict({"fl_algorithm": args.fl_algorithm,
                                 "freeze_layers": args.freeze_layers,
                                 "epochs": args.epochs,
                                 "learning_rate": args.lr,
                                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                 "do_lower_case": args.do_lower_case,
                                 "manual_seed": args.seed,
                                 # for ignoring the cache features.
                                 "reprocess_input_data": args.reprocess_input_data,
                                 "overwrite_output_dir": True,
                                 "max_seq_length": args.max_seq_length,
                                 "train_batch_size": args.train_batch_size,
                                 "eval_batch_size": args.eval_batch_size,
                                 "evaluate_during_training": False,  # Disabled for FedAvg.
                                 "evaluate_during_training_steps": args.evaluate_during_training_steps,
                                 "fp16": args.fp16,
                                 "data_file_path": args.data_file_path,
                                 "partition_file_path": args.partition_file_path,
                                 "partition_method": args.partition_method,
                                 "dataset": args.dataset,
                                 "output_dir": args.output_dir,
                                 "fedprox_mu": args.fedprox_mu
                                 })
    model_args.config["num_labels"] = num_labels
    return model_args


def create_model(args, formulation="classification"):
    # create model, tokenizer, and model config (HuggingFace style)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][args.model_type]
    config = config_class.from_pretrained(args.model_name, **args.config)
    model = model_class.from_pretrained(args.model_name, config=config)
    if formulation != "seq2seq":
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name, do_lower_case=args.do_lower_case)
    else:
        tokenizer = [None, None]
        tokenizer[0] = tokenizer_class.from_pretrained(args.model_name)
        tokenizer[1] = tokenizer[0]

    # logging.info(self.model)
    return config, model, tokenizer
