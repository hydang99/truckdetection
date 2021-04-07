# Truck Detective Code

This repository releases the code for Truck Detection Project from Fort Capital. 
The project is processed under Texas Christian University's student group. 


Should you have any query please contact me at [hy.dang@tcu.edu, trang.dao@tcu.edu,, d.dhamo@tcu.edu, b.rules@tcu.edu, minh.d.nguyen@tcu.edu](mailto:hy.dang@tcu.edu).

## Dependencies
- Python 3.8.1
- Torch 1.7.1
- torchvision 0.8.2
- detectron2 0.3
- You can install all requirements package by running `pip install -r requirements.txt`

## Directories
- src: source files;
- res: resource files including,
    - Vocabulary file `vocab.txt`;
    - Pre-trained embeddings of [GloVe](https://github.com/stanfordnlp/GloVe). We used the GloVe obtained from the Twitter corpora which you could download [here](http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip).
- data: datasets consisting of tweets and prices which you could download [here](https://github.com/yumoxu/stocknet-dataset).

## Configurations
Model configurations are listed in `config.yml` where you could set `variant_type` to *hedge, tech, fund* or *discriminative* to get four corresponding model variants, HedgeFundAnalyst, TechincalAnalyst, FundamentalAnalyst or DiscriminativeAnalyst described in the paper. 

Additionally, when you set `variant_type=hedge, alpha=0`, you would acquire IndependentAnalyst without any auxiliary effects. 

## Running

After configuration, use `sh src/run.sh` in your terminal to start model learning and test the model after the training is completed. If you would like to do them separately, simply comment out `exe.train_and_dev()` or `exe.restore_and_test()` in `Main.py`.
