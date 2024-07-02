import argparse
import pickle
from typing import Dict, List, Set, Tuple

import numpy as np
from WordTransformer import InputExample, WordTransformer


class DataVecLoader:
    def __init__(self):

        self.id2word: Dict[int, str] = {}
        self.word2id: Dict[str, int] = {}
        self.wordid2sentids: Dict[int, List[int]] = {}
        self.wordid2instids: Dict[int, List[int]] = {}

        self.id2sent: Dict[int, str] = {}
        self.sent2id: Dict[str, int] = {}
        self.sentid2instids: Dict[int, List[int]] = {}

        self.id2pos: Dict[int, List[int]] = {}

        self.id2inst: Dict[int, Tuple[int, List[int]]] = {}

        self.vecs: List[List[float]] = []

    def preprocess_wic(self, file_path: str):

        wordVocab: Set[str] = set()
        sentVocab: Set[str] = set()

        curr_wordid = 0
        curr_sentid = 0

        with open(file_path) as fp:
            for instance_id, line in enumerate(fp):
                items = line.strip().split("\t")
                word = items[0]
                pos_s_0 = int(items[2])
                pos_e_0 = int(items[3])
                pos_s_1 = int(items[4])
                pos_e_1 = int(items[5])
                sent_0 = items[6]
                sent_1 = items[7]
                label = items[-1]

                for sent, pos_s, pos_e in [
                    (sent_0, pos_s_0, pos_e_0),
                    (sent_1, pos_s_1, pos_e_1),
                ]:
                    if sent not in sentVocab:
                        sentVocab.add(sent)
                        self.id2sent[curr_sentid] = sent
                        self.sent2id[sent] = curr_sentid
                        self.sentid2instids[curr_sentid] = []
                        self.id2pos[curr_sentid] = [pos_s, pos_e]
                        curr_sentid += 1
                    else:
                        self.sentid2instids[self.sent2id[sent]].append(instance_id)

                if word not in wordVocab:
                    wordVocab.add(word)
                    self.id2word[curr_wordid] = word
                    self.word2id[word] = curr_wordid
                    self.wordid2sentids[curr_wordid] = []
                    self.wordid2instids[curr_wordid] = []
                    curr_wordid += 1
                else:
                    self.wordid2sentids[self.word2id[word]].append(self.sent2id[sent_0])
                    self.wordid2sentids[self.word2id[word]].append(self.sent2id[sent_1])
                    self.wordid2sentids[self.word2id[word]] = list(
                        set(self.wordid2sentids[self.word2id[word]])
                    )
                    self.wordid2instids[self.word2id[word]].append(instance_id)

                self.id2inst[instance_id] = (
                    self.word2id[word],
                    [self.sent2id[sent_0], self.sent2id[sent_1]],
                    label,
                )

        assert len(self.sent2id) == len(self.id2sent) == len(sentVocab)
        assert len(self.word2id) == len(self.id2word) == len(wordVocab)

        self._dics = [
            self.id2word,
            self.word2id,
            self.wordid2sentids,
            self.wordid2instids,
            self.id2sent,
            self.sent2id,
            self.sentid2instids,
            self.id2inst,
        ]

    def save_dic(self, data_type: str = "test"):
        names = [
            "id2word",
            "word2id",
            "wordid2sentids",
            "wordid2instids",
            "id2sent",
            "sent2id",
            "sentid2instids",
            "id2inst",
        ]
        for dic, name in zip(self._dics, names):
            with open(f"{name}.{data_type}", "w") as fp:
                for key, value in dic.items():
                    fp.write(f"{key}\t{value}\n")

    def encode(self, is_tuned: bool = True):
        if is_tuned:
            model = WordTransformer("pierluigic/xl-lexeme")
        else:
            model = WordTransformer("xlm-roberta-large")
        model.to("cpu")
        for sent, pos in zip(self.id2sent.values(), self.id2pos.values()):
            example = InputExample(texts=sent, positions=pos)
            vec = model.encode(example, show_progress_bar=False)
            self.vecs.append(vec)

        self.vecs = np.array(self.vecs)


def calculate_instance(dvl, vecs_decomposed):
    N = len(dvl.id2inst)
    D = vecs_decomposed.shape[1]
    vecs_instance = np.zeros([N, D])

    for instance_id in range(N):
        _, sent_ids, label = dvl.id2inst[instance_id]
        sent_id_0, sent_id_1 = sent_ids
        vec_0 = vecs_decomposed[sent_id_0]
        vec_1 = vecs_decomposed[sent_id_1]
        vecs_instance[instance_id] = abs(vec_0 - vec_1)

    return vecs_instance


def _main(path_contextual_scd: str, is_tuned: bool = True):
    dvl = DataVecLoader()
    dvl.preprocess_wic(path_contextual_scd)
    dvl.save_dic("test")
    dvl.encode(is_tuned=is_tuned)

    if is_tuned:
        pickle.dump(dvl, open("dvl_finetuned.pkl", "wb"))
    else:
        pickle.dump(dvl, open("dvl_pretrained.pkl", "wb"))


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_contextual_scd", help="path to contextualised scd dataset"
    )

    args = parser.parse_args()
    _main(args.path_contextual_scd)
    _main(args.path_contextual_scd, False)


if __name__ == "__main__":
    cli_main()
