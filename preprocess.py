from logging import getLogger, StreamHandler, Formatter, DEBUG
from pathlib import Path
from experimental_tools.preprocess.ner_mturk import NerMturkPreprocessor

logger = getLogger(__name__)
handler = StreamHandler()
format = Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
handler.setFormatter(format)
logger.addHandler(handler)


def main():
    base_dir = Path.cwd()
    embedding_file = base_dir.joinpath("data/glove.6B.300d.txt")

    NerMturkPreprocessor(
        embedding_file,
        data_dir=base_dir.joinpath("data/ner-mturk"),
        save_dir=base_dir.joinpath("data/ner-mturk/preprocessed"),
    )()


if __name__ == "__main__":
    main()
