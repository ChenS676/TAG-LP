# TAG-LP

## Usage 
demo of TAG-LP multi-gpu 

python -m torch.distributed.launch loaders/demo.py --do_train


demo of TAG-LP single-gpu 

python loaders/demo.py --do_train

More details please refer to https://pytorch.org/docs/stable/elastic/run.html