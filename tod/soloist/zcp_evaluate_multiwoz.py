#!/bin/env python
import sys
sys.path.append("./paraphrase/")
import os
import argparse
import logging
import torch
import transformers
import nltk

from utils import setup_logging, pull_model  # noqa:E402
from data.utils import BeliefParser, wrap_dataset_with_cache  # noqa: E402
from data import load_dataset  # noqa: E402
from data.evaluation.multiwoz import MultiWozEvaluator, compute_bleu_remove_reference  # noqa: E402
from generate import generate_predictions  # noqa:E402
from evaluation_utils import compute_delexicalized_bleu  # noqa:E402



def parse_predictions(dataset, filename):
    gts, bfs, rs = [], [], []
    delexrs, delexgts = [], []
    parser = BeliefParser()

    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('GT:'):
                gts.append(line[len('GT:'):])
            elif line.startswith('GTD:'):
                delexgts.append(line[len('GTD:'):])
            elif line.startswith('BF:'):
                bf = line[len('BF:'):]
                bf = parser(bf)
                assert bf is not None
                bfs.append(bf)
            elif line.startswith('RD:'):
                delexrs.append(line[len('RD:'):])
            elif line.startswith('R:'):
                r = line[len('R:'):]
                rs.append(r)
    # assert len(gts) == len(bfs) == len(rs) == len(delexrs) == len(delexgts)

    ## adaption for those pure NLG methods like sclstm.
    if len(rs)==0:
        rs=delexrs
        gts=rs
        bfs=rs # because we do not record the belief states related values in pure NLG components.

        
    return rs, bfs, gts, delexrs, delexgts


def update_delex_response(old_lines, delex_list):
    d_index = 0
    strings = ""
    for line in old_lines:
        if "RD:" in line:
            strings += "RD:{}\n".format(delex_list[d_index])
            d_index += 1
        else:
            strings += line
    return strings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--file', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--save_file', default=None)
    parser.add_argument('--inference_model',type=str, default="data/rettig_model")
    parser.add_argument('--dataset', default='multiwoz-2.1-test')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--add_map',default=0,type=int)
    parser.add_argument('--cuda_num',default=7,type=int)
    parser.add_argument('--gbias',default=0,type=int)
    parser.add_argument('--bp',default=0,type=int)
    args = parser.parse_args()
    if args.resume is not None and args.model is None:
        args.model = f'wandb:{args.resume}'
    assert args.model is not None or args.file is not None

    # # Update punkt
    # nltk.download('punkt')

    setup_logging()
    logger = logging.getLogger()
    if args.resume:
        import wandb
        # Resume run and fill metrics
        os.environ.pop('WANDB_NAME', None)
        wandb.init(resume=args.resume)
    elif args.wandb:
        import wandb
        # It is an artifact
        # Start a new evaluate run
        wandb.init(job_type='evaluation')
    else:
        wandb = None

    dataset = load_dataset(args.dataset, use_goal=True)
    dataset = wrap_dataset_with_cache(dataset)

    if args.file is None or not os.path.exists(args.file):
        args.model = pull_model(args.model)

    if args.file is not None:
        path = args.file
        if not os.path.exists(path):
            path = os.path.join(args.model, args.file)
        responses, beliefs, gold_responses, delex_responses, delex_gold_responses = parse_predictions(dataset, path)
    else:
        logger.info('generating responses')
        pipeline = transformers.pipeline('soloist-conversational', args.model, device=args.cuda_num if torch.cuda.is_available() else -1)
        responses, beliefs, gold_responses, delex_responses, delex_gold_responses = \
            generate_predictions(pipeline, dataset, os.path.join(wandb.run.dir if wandb and wandb.run else '.', args.save_file))

    if args.add_map == 1:
        from paraphrase.inference import Inference
        logger.info("we need to make map with rettig")
        transformModel = Inference(
            args.inference_model, cuda_num=args.cuda_num,gbias=args.gbias,bp=args.bp)
        logger.info("init DONEEEE.")
        # print("old:",type(delex_responses))
        # print("old sentence example:", delex_responses)
        # print(f"==={delex_responses}===")
        delex_responses = transformModel.inference(delex_responses)
        # print(f"---{delex_responses}---")
        # print("new sentence example:", delex_responses)
        # print("new:",type(delex_responses))
        if args.file is not None:
            with open(args.file, 'r') as f:
                original_lines = f.readlines()
            ## now we replace the original delexicalized result into new results and save it.
            new_lines_string = update_delex_response(
                original_lines, delex_responses)
            with open(args.save_file, 'w') as f:
                f.write(new_lines_string)
    else:
        logger.info("do not needed to map delex_responses...")
    logger.info('evaluation started')
    evaluator = MultiWozEvaluator(dataset, is_multiwoz_eval=True, logger=logger)
    success, matches, domain_results = evaluator.evaluate(beliefs, delex_responses, progressbar=True)
    logger.info('evaluation finished')
    logger.info(f'match: {matches:.4f}, success: {success:.4f}')
    logger.info('computing bleu')
    if wandb and wandb.run:
        wandb.run.summary.update(dict(
            test_inform=matches,
            test_success=success,
        ))
    if dataset.lexicalizer is not None:
        bleu = compute_bleu_remove_reference(responses, gold_responses)
        logger.info(f'test bleu: {bleu:.4f}')
        if wandb and wandb.run:
            wandb.run.summary.update(dict(test_bleu=bleu))

    delex_bleu = compute_delexicalized_bleu(delex_responses, delex_gold_responses)
    logger.info(f'test delex bleu: {delex_bleu:.4f}')
    if wandb and wandb.run:
        wandb.run.summary.update(dict(test_delex_bleu=delex_bleu))

