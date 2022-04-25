import sys
import torch

# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def evaluate(model, tokenizer):
    def prepare(params, samples):
        return
    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )
        for k in batch:
            batch[k] = batch[k].to('cuda')
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
        return outputs.last_hidden_state[:, 0].cpu() # unpooled [CLS] output in BERT


    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}

    se = senteval.engine.SE(params, batcher, prepare)
    tasks = ['STSBenchmark']
    results = se.eval(tasks)
    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    stsb_align = results['STSBenchmark']['dev']['align_loss']
    stsb_uniform = results['STSBenchmark']['dev']['uniform_loss']

    metrics = {"eval_stsb_spearman": stsb_spearman,
                "eval_stsb_align": stsb_align,
                "eval_stsb_uniform": stsb_uniform}

    return metrics


def inf_train_gen(trainloader):
    while True:
        for batch in trainloader:
            yield batch