import argparse
import random

random.seed(12345)

from vec_utils import *
from visualise_contextual import *
from visualise_temporal import *
from visualise_utils import *


def main(
    path_contextual_finetuned: str,
    path_contextual_pretrained: str,
    output: str = "test",
):

    visualise_instance(path_contextual_finetuned, output=f"{output}_ft")
    visualise_instance(path_contextual_finetuned, conduct_ica=False, output=f"{output}_ft")
    visualise_instance(path_contextual_pretrained, output=f"{output}_pre")
    visualise_instance(path_contextual_pretrained, conduct_ica=False, output=f"{output}_pre")

    visualise_exp_ratio_all(path_contextual_finetuned, output=f"{output}_ft")
    visualise_exp_ratio_all(path_contextual_pretrained, output=f"{output}_pre")

    visualise_roc(path_contextual_finetuned, output=f"{output}_ft")
    visualise_roc(path_contextual_pretrained, output=f"{output}_pre")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_contextual_finetuned", help="path to contextual scd dvl (fine-tuned)"
    )
    parser.add_argument(
        "--path_contextual_pretrained", help="path to contextual scd dvl (pre-trained)"
    )
    parser.add_argument(
        "--output", default="test", help="output name"
    )
    args = parser.parse_args()
    main(
        args.path_contextual_finetuned,
        args.path_contextual_pretrained,
        args.output,
    )


if __name__ == "__main__":
    cli_main()
