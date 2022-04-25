# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging

import torch
from torch.nn.functional import normalize

from scipy.stats import spearmanr, pearsonr

from senteval.utils import cosine
from senteval.sick import SICKEval


# Question: normalization before evaluation?

def align_loss(x, y, alpha=2):
    if min(x.shape)==0 or min(y.shape)==0:
        return 0
    x = normalize(x, p=2, dim=1)
    y = normalize(y, p=2, dim=1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    # pdist: Computes the p-norm distance between every pair of row vectors in the input
    if min(x.shape) == 0:
        return 0
    x = normalize(x, p=2, dim=1)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def run(self, params, batcher):
        results = {}
        all_sys_scores = []
        all_gs_scores = []
        ################# newly added
        all_loss_align = []
        all_loss_uniform = []
        #################
        for dataset in self.datasets:
            loss_align = []
            loss_uniform = []
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]
                batch_gs_scores = gs_scores[ii:ii + params.batch_size]  # newly added

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    ################# newly added
                    pos_indices = [i for i in range(len(batch_gs_scores)) if batch_gs_scores[i] >= 4.0]
                    enc1_pos = enc1[pos_indices]
                    enc2_pos = enc2[pos_indices]
                    loss1 = align_loss(enc1_pos, enc2_pos)
                    loss2 = uniform_loss(torch.cat((enc1, enc2), dim=0))
                    loss_align.append(loss1)
                    loss_uniform.append(loss2)
                    #################

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)

            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)
            all_loss_align.extend(loss_align)
            all_loss_uniform.extend(loss_uniform)

            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores),
                                'align_loss': np.mean(loss_align, dtype='float64'),  # newly added
                                'uniform_loss': np.mean(loss_uniform, dtype='float64')}  # newly added
            logging.debug('%s : pearson = %.4f, spearman = %.4f, align_loss = %.4f, uniform_loss = %.4f' %
                          (dataset, results[dataset]['pearson'][0],
                           results[dataset]['spearman'][0], results[dataset]['align_loss'],
                           results[dataset]['uniform_loss']))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])
        list_align = np.array([results[dset]['align_loss'] for
                            dset in results.keys()])
        list_uniform = np.array([results[dset]['uniform_loss'] for
                               dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        avg_align = np.average(list_align)
        avg_uniform = np.average(list_uniform)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)
        wavg_align = np.average(list_align, weights=weights)
        wavg_uniform = np.average(list_uniform, weights=weights)
        all_pearson = pearsonr(all_sys_scores, all_gs_scores)
        all_spearman = spearmanr(all_sys_scores, all_gs_scores)
        all_align = np.mean(all_loss_align)
        all_uniform = np.mean(all_loss_uniform)
        results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman},
                          'align_loss':{'all': all_align,
                                       'mean': avg_align,
                                       'wmean': wavg_align},
                          'uniform_loss':{'all': all_uniform,
                                       'mean': avg_uniform,
                                       'wmean': wavg_uniform}}
        logging.debug('ALL : Pearson = %.4f, \
            Spearman = %.4f, align_loss = %.4f, uniform_loss = %.4f' % (all_pearson[0], all_spearman[0], all_align, all_uniform))
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f, align_loss = %.4f, uniform_loss = %.4f' % (wavg_pearson, wavg_spearman, wavg_align, wavg_uniform))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f, align_loss = %.4f, uniform_loss = %.4f\n' % (avg_pearson, avg_spearman, avg_align, avg_uniform))

        return results


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())    # sentence1
                sick_data['X_B'].append(text[6].split())    # sentence2
                sick_data['y'].append(text[4])  # scores

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class STSBenchmarkFinetune(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data
        
class SICKRelatednessEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())  # sentence 1
                    sick_data['X_B'].append(text[2].split())  # sentence 2
                    sick_data['y'].append(text[3])  # relatedness_score

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])
