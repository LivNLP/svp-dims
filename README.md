# svp-dims
Code for "Investigating the Contextualised Word Embedding Dimensions Specified for Contextual and Temporal Semantic Changes"

## Data
 - [Word-in-Context](https://pilehvar.github.io/wic/)
 - [XL-WiC](https://pilehvar.github.io/xlwic/)
 - [MCL-WiC](https://github.com/SapienzaNLP/mcl-wic)
 - [AM2iCo](https://github.com/cambridgeltl/AM2iCo)
 - [SemEval-2020 Task 1](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/)

## Setup
```
pip3 install -r requirements.txt

git clone https://github.com/pierluigic/xl-lexeme.git
cd xl-lexeme
pip3 install .
```

## Calcualte vectors
### Contextual
```
bash process_wic.sh

cd src
python3 preprocess_wic_testset.py \
    --wic_test_data PATH_TO_TEST_DATA \
    --wic_test_gold PATH_TO_TEST_GOLD

python3 datavecloader.py \
    --path_contextual_scd PATH_TO_PROCESSED_DATA
```

### Temporal
```
cd src
python3 prepare_temporal_vecs.py \
    --file_path PATH_TO_SEMEVAL_TOKEN \
    --lemma_path PATH_TO_SEMEVAL_LEMMA \
    --target_words_list PATH_TO_TARGET_WORDS \
    --lang en/la/de/sw \
    --output_name PATH_TO_RESULT_PKL

```

## Visualise
```
python3 main_cont.py \
    --path_contextual_finetuned PATH_DVL_FINETUNED \
    --path_contextual_pretrained PATH_DVL_PRETRAINED \
    --output OUTPUT_NAME # e.g. DATA_LANG

python3 main_temp.py \
    --path_temporal_word2label PATH_WORD2LABEL \
    --path_temporal_word2grade PATH_WORD2GRADE \
    --path_temporal_finetuned_t1 PATH_FINETUNED_t1 \
    --path_temporal_finetuned_t2 PATH_FINETUNED_t2 \
    --path_temporal_pretrained_t1 PATH_PRETRAINED_t1 \
    --path_temporal_pretrained_t2 PATH_PRETRAINED_t2 \
    --output OUTPUT_NAME # e.g. LANG

```
