# TAG-LP

## Usage 
demo of TAG-LP multi-gpu 

python -m torch.distributed.launch loaders/demo.py --do_train


demo of TAG-LP single-gpu 

python loaders/demo.py --do_train

More details please refer to https://pytorch.org/docs/stable/elastic/run.html

# TODO 
- https://wandb.ai/wandb_fc/articles/reports/Monitor-Improve-GPU-Usage-for-Model-Training--Vmlldzo1NDQzNjM3#:~:text=Try%20increasing%20your%20batch%20size&text=Gradients%20for%20a%20batch%20are,increase%20the%20speed%20of%20calculation.
- multi-gpu distributed training principle and implementation
- double check the evaluation method in KG, compare it with others 1. tangji liang lecture, 2. galkin new paper
-  add more evaluation metrics