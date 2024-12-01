import argparse
import random

random.seed(12345)

from vec_utils import *
from visualise_contextual import *
from visualise_temporal import *
from visualise_utils import *


def main(
    path_temporal_word2label: str,
    path_temporal_word2grade: str,
    path_temporal_finetuned_t1: str,
    path_temporal_finetuned_t2: str,
    path_temporal_pretrained_t1: str,
    path_temporal_pretrained_t2: str,
    output: str = "test",
):

    evaluate_scd(
        path_temporal_word2label,
        path_temporal_word2grade,
        path_temporal_finetuned_t1,
        path_temporal_finetuned_t2,
        output=f"{output}_ft",
    )

    evaluate_scd(
        path_temporal_word2label,
        path_temporal_word2grade,
        path_temporal_pretrained_t1,
        path_temporal_pretrained_t2,
        output=f"{output}_pre",
    )

    visualise_scd_roc(
        path_temporal_word2label,
        path_temporal_word2grade,
        path_temporal_finetuned_t1,
        path_temporal_finetuned_t2,
        output=f"{output}_ft",
    )

    visualise_scd_roc(
        path_temporal_word2label,
        path_temporal_word2grade,
        path_temporal_pretrained_t1,
        path_temporal_pretrained_t2,
        output=f"{output}_pre",
    )


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_temporal_word2label", help="path to temporal scd task, binary.txt"
    )
    parser.add_argument(
        "--path_temporal_word2grade", help="path to temporal scd task, graded.txt"
    )
    parser.add_argument(
        "--path_temporal_finetuned_t1", help="path to temporal scd dvl (fine-tuned)"
    )
    parser.add_argument(
        "--path_temporal_finetuned_t2", help="path to temporal scd dvl (fine-tuned)"
    )
    parser.add_argument(
        "--path_temporal_pretrained_t1", help="path to temporal scd dvl (pre-trained)"
    )
    parser.add_argument(
        "--path_temporal_pretrained_t2", help="path to temporal scd dvl (pre-trained)"
    )
    parser.add_argument(
        "--output", default="test", help="output name"
    )
    args = parser.parse_args()
    main(
        args.path_temporal_word2label,
        args.path_temporal_word2grade,
        args.path_temporal_finetuned_t1,
        args.path_temporal_finetuned_t2,
        args.path_temporal_pretrained_t1,
        args.path_temporal_pretrained_t2,
        args.output,
    )


if __name__ == "__main__":
    cli_main()
