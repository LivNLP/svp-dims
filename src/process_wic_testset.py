import argparse


def get_positions(sentence: str, target_word_id: int):
    curr_word_id = 0
    curr_char_id = 0
    positions = []
    for word in sentence.split():
        if curr_word_id == target_word_id:
            positions.append(curr_char_id)
            positions.append(curr_char_id + len(word))
            break
        curr_word_id += 1
        curr_char_id += len(word) + 1

    return positions


def process_wic_testset(wic_test_data: str, wic_test_gold: str):

    lines_data = []
    with open(wic_test_data) as fp:
        for line in fp:
            lines_data.append(line.strip())

    lines_gold = []
    with open(wic_test_gold) as fp:
        for line in fp:
            lines_gold.append(line.strip())

    with open("wic_test_en.txt", "w") as fp:
        for line_data, line_gold in zip(lines_data, lines_gold):
            label = 1 if line_gold == "T" else 0
            word, pos, word_ids_str, sent_0, sent_1 = line_data.strip().split("\t")
            word_id_0_str, word_id_1_str = word_ids_str.split("-")
            word_id_0 = int(word_id_0_str)
            word_id_1 = int(word_id_1_str)
            positions_0 = get_positions(sent_0, word_id_0)
            positions_1 = get_positions(sent_1, word_id_1)

            line_test = [
                word,
                pos,
                str(positions_0[0]),
                str(positions_0[1]),
                str(positions_1[0]),
                str(positions_1[1]),
                sent_0,
                sent_1,
                str(label),
            ]

            _line_test = "\t".join(line_test)
            fp.write(f"{_line_test}\n")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wic_test_data")
    parser.add_argument("--wic_test_gold")
    args = parser.parse_args()
    process_wic_testset(args.wic_test_data, args.wic_test_gold)


if __name__ == "__main__":
    cli_main()
