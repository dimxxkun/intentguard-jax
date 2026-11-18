"""Train WordPiece only on the training role; use the Transformers tokenizer contract."""

from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


def fit_tokenizer(texts, directory, vocab_size=4096):
    backend = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    )
    backend.train_from_iterator(texts, trainer=trainer)
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    tokenizer.save_pretrained(str(directory))
    return tokenizer


def load_tokenizer(directory):
    return PreTrainedTokenizerFast.from_pretrained(str(Path(directory)), local_files_only=True)


def encode(tokenizer, texts, max_length):
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="np",
        return_token_type_ids=False,
    )
