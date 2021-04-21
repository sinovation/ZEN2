# ZEN 2.0

ZEN 2.0 is based on the architecture of ZEN 1.0 with an update and adaptation from three aspects, after which the model is upgraded into the same scale of BERT-large and applied to different languages (i.e., Chinese and Arabic).   First,  we  refine  the  representations  of n-grams by applying weights to the n-gram representations when integrating them into the character encoder.  Second, in the training stage, we mask n-grams/words, rather than characters, in the input text of the character encoders.  Third, we utilize relative positional encoding for the character encoder to model direction and distance information from the input text.  The structure of ZEN 2.0 is illustrated in the figure below.
　

![ZEN_model](http://zen.chuangxin.com/front/assets/zen2.png)

　

## Quick tour of pre-training and fine-tune using ZEN 2.0

The library comprises several example scripts for conducting [**Chinese NLP tasks**](/datasets):

- `run_pre_train.py`: an example pre-training ZEN 2.0
- `run_sequence_level_classification.py`: an example fine-tuning ZEN 2.0 on DC, SA, SPM and NLI tasks (*sequence-level classification*)
- `run_token_level_classification.py`: an example fine-tuning ZEN 2.0 on CWS, POS and NER tasks (*token-level classification*)
- `run_mrc.py`: an example fine-tuning ZEN 2.0 on MRC tasks

[**Examples**](/examples) of pre-training and fine-tune using ZEN 2.0.


## Contact information

For help or issues using ZEN 2.0, please submit a GitHub issue.

For personal communication related to ZEN 2.0, please contact chenguimin(`chenguimin@chuangxin.com`).

