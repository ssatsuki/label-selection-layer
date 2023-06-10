import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from conlleval import conlleval

# from experimental_tools.preprocess.ner_mturk import NerMturkPreprocessor

# NOTE: INDEX2LABEL = NerMturkPreprocessor.INDEX2LABEL
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


# https://github.com/fmpr/CrowdLayer/blob/master/demo-conll-ner-mturk.ipynb
# modified score func and rename it flatten
def flatten(y, preds):
    # NOTE: Get the index of the first character by np.where(seq > 0)[0][0].
    start_indices = [np.where(seq > 0)[0][0] for seq in y]
    y_ = [seq[start_idx:] for seq, start_idx in zip(y, start_indices)]
    y_preds = [pred[start_idx:] for pred, start_idx in zip(preds, start_indices)]
    y_flatten = [c for row in y_ for c in row]
    y_preds_flatten = [c for row in y_preds for c in row]
    return y_flatten, y_preds_flatten


# https://github.com/fmpr/CrowdLayer/blob/master/demo-conll-ner-mturk.ipynb
def read_conll(filename):
    raw = open(filename, "r").readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(" ")
        point.append(stripped_line)
        if line == "\n":
            if len(point[:-1]) > 0:
                all_x.append(point[:-1])
            point = []
    all_x = all_x
    return all_x


def main(args):
    base_pred_dir = Path.cwd().joinpath("multirun").joinpath(args.strdatetime)

    ner_dir = Path.cwd().joinpath("data/ner-mturk/")
    all_test = read_conll(ner_dir.joinpath("testset.txt"))
    X_test = [[c[0] for c in x] for x in all_test]
    y_test = [[c[1] for c in y] for y in all_test]

    results = list()
    for i in range(100):
        parend_pred_dir = base_pred_dir.joinpath(str(i))
        try:
            for pred_dir in parend_pred_dir.iterdir():
                if pred_dir.suffix == ".npy":
                    print(pred_dir)
                    preds = np.load(pred_dir).argmax(axis=2)
                    break
            preds_test = []
            for j in range(len(preds)):
                row = preds[j][-len(y_test[j]):]
                # NOTE: unify index which represent 'O'.
                row[np.where(row == 0)] = 1
                preds_test.append(row)
            preds_test = [list(map(lambda x: INDEX2LABEL[x], y_)) for y_ in preds_test]
            # print(preds_test)
            res = conlleval(preds_test, y_test, X_test, f"r_test_{i:02}.txt")
            print(res)
            results.append(res)
        except:
            break

    df = pd.DataFrame(results)
    print(df)
    print(df.mean())
    print(df.std())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="print each experimental result.",
    )
    parser.add_argument("strdatetime", help="YYYY-MM-DD/hh-mm-ss", type=str)
    args = parser.parse_args()
    main(args)
