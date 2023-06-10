""" This is a rewritten code based on the following code:
  - https://github.com/fmpr/CrowdLayer/blob/master/demo-conll-ner-mturk.ipynb
"""
import pickle
from logging import DEBUG, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import Any, Dict, Final, List
from datetime import datetime

import numpy as np
import pandas as pd
from crowdkit.aggregation import DawidSkene
from tqdm import tqdm

from ..typing import Annotations, Features, Labels, OrgData

logger = getLogger(__name__)
handler = StreamHandler()
format = Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
handler.setFormatter(format)
logger.addHandler(handler)


class NerMturkPreprocessor:
    N_ANNOTATORS: Final[int] = 47
    N_WORDS: Final[int] = 18179  # == len(word2ind)
    EMBEDDING_DIM: Final[int] = 300

    LABEL2INDEX = {
        "O": 1,
        "B-ORG": 2,
        "I-ORG": 3,
        "B-PER": 4,
        "I-PER": 5,
        "B-LOC": 6,
        "I-LOC": 7,
        "B-MISC": 8,
        "I-MISC": 9,
    }

    INDEX2LABEL = {
        0: "O",
        1: "O",
        2: "B-ORG",
        3: "I-ORG",
        4: "B-PER",
        5: "I-PER",
        6: "B-LOC",
        7: "I-LOC",
        8: "B-MISC",
        9: "I-MISC",
    }

    MAX_LABEL_INDEX = max(LABEL2INDEX.values()) + 1
    N_CLASSES = len(LABEL2INDEX) + 1

    def __init__(self, embedding_file: Path, data_dir: Path, save_dir: Path):
        if not isinstance(embedding_file, Path):
            raise TypeError('Required "Path" type data as embedding_file.')
        self.embedding_file = embedding_file
        logger.info(f"Set {self.embedding_file} as embedding_file.")

        self.data_dir = data_dir
        logger.info(f"Set {self.data_dir} as data_dir.")

        self.save_dir = save_dir
        logger.info(f"Set {self.save_dir} as save_dir.")

    def __call__(self, is_forced: bool = False) -> None:
        if (
            not is_forced
            and self.save_dir.joinpath("ner_mturk_preprocess_finished.pkl").exists()
        ):
            logger.info("Skip NER Mturk data preprocessing.")
            return None

        logger.info("Run NER Mturk data preprocessing.")
        embeddings_index = self.load_embeddings()
        org_data: Dict[str, OrgData] = {}
        for filename in ["answers", "mv", "ground_truth", "testset"]:
            org_data[filename] = self.read_conll(filename + ".txt")
        org_data["docs"] = org_data["ground_truth"] + org_data["testset"]
        logger.info(f"Size of all_docs data is {len(org_data['docs'])}")

        # extract words
        X_train = self.extract_words(org_data["answers"])
        X_test = self.extract_words(org_data["testset"])
        X_all = self.extract_words(org_data["docs"])

        # extract label annotations
        y_answers = self.extract_annotations(org_data["answers"])

        # extract labels
        y_mv = self.extract_labels(org_data["mv"])
        y_gt = self.extract_labels(org_data["ground_truth"])
        y_test = self.extract_labels(org_data["testset"])

        all_text = [c for x in X_all for c in x]
        words = list(set(all_text))

        from sklearn.model_selection import train_test_split

        X_train, X_valid, y_train, y_valid, y_answers, _, y_mv, _ = train_test_split(
            X_train, y_gt, y_answers, y_mv, shuffle=True, random_state=42, test_size=0.2
        )

        word2ind = {word: index for index, word in enumerate(words)}
        ind2word = {index: word for index, word in enumerate(words)}
        logger.info(f"Max label index: {self.MAX_LABEL_INDEX}")

        logger.info(f"Length of X_train is {len(X_train)}")
        logger.info(f"Length of X_valid is {len(X_valid)}")
        logger.info(f"Length of y_train is {len(y_train)}")
        logger.info(f"Length of y_valid is {len(y_valid)}")
        logger.info(f"Length of y_answers is {len(y_answers)}")
        logger.info(f"Length of y_mv is {len(y_mv)}")

        maxlen = max([len(x) for x in X_all])
        logger.info(f"Maximum sequence length: {maxlen}")

        embeddings_matrix = self.load_word_vectors(embeddings_index, word2ind)

        y_answers_enc = []
        for r in range(self.N_ANNOTATORS):
            annot_answers = []
            for i in range(len(y_answers)):
                seq = []
                for j in range(len(y_answers[i])):
                    enc = -1
                    if y_answers[i][j][r] != "?":
                        enc = self.LABEL2INDEX[y_answers[i][j][r]]
                    seq.append(enc)
                annot_answers.append(seq)
            y_answers_enc.append(annot_answers)

        # pad sequences
        X_train_enc = self.pad_sequences(
            [[word2ind[c] for c in x] for x in X_train], maxlen=maxlen
        )
        X_valid_enc = self.pad_sequences(
            [[word2ind[c] for c in x] for x in X_valid], maxlen=maxlen
        )
        y_train_enc = self.pad_sequences(
            self.pad_and_encode(y_train, maxlen), maxlen=maxlen
        )
        y_valid_enc = self.pad_sequences(
            self.pad_and_encode(y_valid, maxlen), maxlen=maxlen
        )
        X_test_enc = self.pad_sequences(
            [[word2ind[c] for c in x] for x in X_test], maxlen=maxlen
        )
        y_test_enc = self.pad_sequences(
            self.pad_and_encode(y_test, maxlen), maxlen=maxlen
        )
        y_mv_enc = self.pad_sequences(self.pad_and_encode(y_mv, maxlen), maxlen=maxlen)

        y_answers_enc_ = np.transpose(
            np.array(
                [
                    self.pad_sequences(y_answers_enc[r], maxlen)
                    for r in range(self.N_ANNOTATORS)
                ]
            ),
            [1, 2, 0],
        )

        logger.info(f"Shape of X_train_enc is {X_train_enc.shape}.")
        logger.info(f"Shape of X_valid_enc is {X_valid_enc.shape}.")
        logger.info(f"Shape of X_test_enc is {X_test_enc.shape}.")
        logger.info(f"Shape of y_train_enc is {y_train_enc.shape}.")
        logger.info(f"Shape of y_valid_enc is {y_valid_enc.shape}.")
        logger.info(f"Shape of y_test_enc is {y_test_enc.shape}.")
        logger.info(f"Shape of answers is {y_answers_enc_.shape}.")
        logger.info(f"The number of classes is {self.N_CLASSES}.")

        self.save_dir.mkdir(exist_ok=True, parents=True)

        with self.save_dir.joinpath("ind2word.pkl").open(mode="wb") as f:
            pickle.dump(ind2word, f)

        self.save("ner_mturk_encoded_train_features.npy", X_train_enc)
        self.save("ner_mturk_encoded_valid_features.npy", X_valid_enc)
        self.save("ner_mturk_encoded_test_features.npy", X_test_enc)
        self.save("ner_mturk_train_labels_gt.npy", y_train_enc)
        self.save("ner_mturk_valid_labels_gt.npy", y_valid_enc)
        self.save("ner_mturk_test_labels.npy", y_test_enc)
        self.save("ner_mturk_train_labels_answers.npy", y_answers_enc_)
        self.save("ner_mturk_train_labels_mv.npy", y_mv_enc)
        self.save("ner_mturk_embeddings.npy", embeddings_matrix)

        answers_df = self.convert_crowdkit_format(y_answers_enc_)
        y_ds_enc = self.get_dawid_skene(y_train_enc, answers_df, maxlen)
        self.save("ner_mturk_train_labels_ds.npy", y_ds_enc)

        with self.save_dir.joinpath("ner_mturk_preprocess_finished.pkl").open(
            mode="wb"
        ) as f:
            pickle.dump({"created_at": datetime.now()}, f)

    def load_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings_index: Dict[str, np.ndarray] = {}
        with self.embedding_file.open(mode="r") as f:
            for line in tqdm(f):
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype="float32")
                embeddings_index[word] = coefs
        logger.info(f"Loaded {len(embeddings_index)} word vectors.")
        return embeddings_index

    def load_word_vectors(
        self,
        embeddings_index: Dict[str, np.ndarray],
        word2ind: Dict[str, int],
    ) -> np.ndarray:
        embeddings_matrix = np.zeros((self.N_WORDS, self.EMBEDDING_DIM))
        for word, i in word2ind.items():
            embeddings_vector = embeddings_index.get(word)
            if embeddings_vector is not None:
                embeddings_matrix[i] = embeddings_vector
        return embeddings_matrix

    @classmethod
    def encode(cls, x):
        result = np.zeros(cls.MAX_LABEL_INDEX)
        result[x] = 1
        return result

    @classmethod
    def pad_and_encode(cls, y: Labels, maxlen: int) -> List[List[int]]:
        padded_y = [
            [0] * (maxlen - len(seq)) + [cls.LABEL2INDEX[c] for c in seq]
            for seq in y  # noqa
        ]
        encoded_y = [[cls.encode(c) for c in seq] for seq in padded_y]
        return encoded_y

    def read_conll(self, filename: str) -> OrgData:
        logger.info(f"Start loading {filename}.")
        data = []
        point = []
        with self.data_dir.joinpath(filename).open("r") as f:
            for line in f.readlines():
                stripped_line = line.strip().split(" ")
                point.append(stripped_line)
                if line == "\n":
                    if len(point[:-1]) > 0:
                        data.append(point[:-1])
                    point = []
        logger.info(f"Size of {filename} data is {len(data)}")
        return data

    @staticmethod
    def extract_words(data: OrgData) -> Features:
        return [[c[0] for c in x] for x in data]

    @staticmethod
    def extract_annotations(data: OrgData) -> Annotations:
        return [[c[1:] for c in y] for y in data]

    @staticmethod
    def extract_labels(data: OrgData) -> Labels:
        return [[c[1] for c in y] for y in data]

    def save(self, filename: str, data: Any):
        np.save(self.save_dir.joinpath(filename), data)

    @staticmethod
    def pad_sequences(
        sequences,
        maxlen=None,
        dtype="int32",
        padding="pre",
        truncating="pre",
        value=0.0,
    ):
        if not hasattr(sequences, "__len__"):
            raise ValueError("`sequences` must be iterable.")
        num_samples = len(sequences)

        lengths = []
        sample_shape = ()
        flag = True

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.

        for x in sequences:
            try:
                lengths.append(len(x))
                if flag and len(x):
                    sample_shape = np.asarray(x).shape[1:]
                    flag = False
            except TypeError:
                raise ValueError(
                    "`sequences` must be a list of iterables. "
                    "Found non-iterable: " + str(x)
                )

        if maxlen is None:
            maxlen = np.max(lengths)

        is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
            dtype, np.unicode_
        )
        if isinstance(value, str) and dtype != object and not is_dtype_str:
            raise ValueError(
                f"`dtype` {dtype} is not compatible with `value`'s type: "
                f"{type(value)}\n"
                "You should set `dtype=object` for variable length strings."
            )

        x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
        for idx, s in enumerate(sequences):
            if not len(s):
                continue  # empty list/array was found
            if truncating == "pre":
                trunc = s[-maxlen:]
            elif truncating == "post":
                trunc = s[:maxlen]
            else:
                raise ValueError(f'Truncating type "{truncating}" ' "not understood.")

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError(
                    "Shape of sample %s of sequence at position %s "
                    "is different from expected shape %s"
                    % (trunc.shape[1:], idx, sample_shape)
                )

            if padding == "post":
                x[idx, : len(trunc)] = trunc
            elif padding == "pre":
                x[idx, -len(trunc) :] = trunc  # noqa
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x

    @staticmethod
    def convert_crowdkit_format(answers: np.ndarray) -> pd.DataFrame:
        n_seqs, maxlen, n_annotators = answers.shape
        logger.info(f"Shape of answers is {answers.shape}")
        dfs = list()
        for i in range(n_seqs):
            for j in range(maxlen):
                for w in range(n_annotators):
                    v = answers[i, j, w]
                    if v != -1:
                        t = i * maxlen + j
                        dfs.append([w, t, v])
        return pd.DataFrame(dfs, columns=["worker", "task", "label"])

    @staticmethod
    def get_dawid_skene(
        y: np.ndarray,
        answers_df: pd.DataFrame,
        maxlen: int,
        n_iter: int = 1_000,
    ) -> pd.DataFrame:
        logger.info("Start Dawid and Skene Estimation.")
        ds = DawidSkene(n_iter=n_iter).fit_predict(answers_df)
        y_ds = np.zeros_like(y) * (-1)
        for t, label in ds.iteritems():
            i = t // maxlen
            j = t % maxlen
            y_ds[i, j, label] = 1
        return y_ds
