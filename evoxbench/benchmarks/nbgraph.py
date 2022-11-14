""" NASBench 101 Benchmark """
import copy
import json
from ast import literal_eval
import hashlib
import itertools
import os
import random
from collections import OrderedDict
from pathlib import Path
import numpy as np
from numpy import ndarray
from typing import Callable, Sequence, cast, List, AnyStr, Any, Tuple, Union, Set

from evoxbench.modules import SearchSpace, Evaluator, Benchmark
from nasbenchgraph.models import NASBenchGraphResult  # has to be imported after the init method

__all__ = ['NASBenchGraphSearchSpace', 'NASBenchGraphEvaluator', 'NASBenchGraphBenchmark']

HASH = {'conv3x3-bn-relu': 0, 'conv1x1-bn-relu': 1, 'maxpool3x3': 2}


def get_path(name):
    print(name)
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "nbgraph" / name)


class NASBenchGraphSearchSpace(SearchSpace):
    """
        NASBenchGraph API need to be first installed following the official instructions fromb
        https://github.com/google-research/nasbench
    """

    def __init__(self, **kwargs
                 ):
        super().__init__(**kwargs)

        self.n_var = 9
        self.lb = [0] * self.n_var
        self.ub = [5, 1, 5, 1, 1, 5, 1, 1, 1]

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    @property
    def name(self):
        return 'NASBenchGraphSearchSpace'

    def _sample(self, phenotype=True):
        tmp = [random.randint(self.lb[i], self.ub[i]) for i in range(len(self.lb))]
        tmp = [tuple(tmp[:2]), tuple(tmp[2:5]), tuple(tmp[5:9])]
        arch = str(tmp)
        return arch if phenotype else self._encode(arch)

    def _encode(self, arch: str):
        # encode architecture phenotype to genotype
        # a sample architecture
        # '64:56:8:24:16:24:48:8'

        arch = literal_eval(arch.replace('(', '').replace(')', ''))
        return np.array(arch)

    def _decode(self, x: ndarray) -> str:
        ans = x.tolist()

        return str([tuple(ans[:2]), tuple(ans[2:5]), tuple(ans[5:9])])

    def visualize(self, arch):
        raise NotImplementedError


class NASBenchGraphEvaluator(Evaluator):
    def __init__(self,
                 objs='err&params&flops&gtx-1080ti-fp32&jetson-nano-fp32',  # objectives to be minimized
                 ):
        super().__init__(objs)


    @property
    def name(self):
        return 'NASBenchGraphEvaluator'

    def evaluate(self, archs, objs=None,
                 true_eval=False  # query the true (mean over three runs) performance
                 ):

        if objs is None:
            objs = self.objs

        batch_stats = []
        results = {i.genotype: i for i in NASBenchGraphResult.objects.filter(genotype__in=archs)}
        for i, arch in enumerate(archs):
            stats = {}
            if arch not in results:
                if 'err' in objs:
                    stats['err'] = 1
                if 'params' in objs:
                    stats['params'] = np.inf
                if 'flops' in objs:
                    stats['flops'] = np.inf
                if 'gtx-1080ti-fp32' in objs:
                    stats['gtx-1080ti-fp32'] = np.inf
                if 'jetson-nano-fp32' in objs:
                    stats['jetson-nano-fp32'] = np.inf
                batch_stats.append(stats)
                continue
            ans = results[arch]
            top1 = ans.test_per
            params = ans.params
            flops = ans.flops
            if 'err' in objs:
                stats['err'] = 1 - top1
            if 'params' in objs:
                stats['params'] = params
            if 'flops' in objs:
                stats['flops'] = flops
            if 'gtx-1080ti-fp32' in objs:
                stats['gtx-1080ti-fp32'] = ans.latency['gtx-1080ti-fp32']
            if 'jetson-nano-fp32' in objs:
                stats['jetson-nano-fp32'] = ans.latency['jetson-nano-fp32']
            batch_stats.append(stats)

        return batch_stats


class NASBenchGraphBenchmark(Benchmark):
    def __init__(self,

                 objs='err&params&flops',  # objectives to be minimized
                 pf_file_path=get_path("asr_pf.json"),  # path to NASBenchGraph Pareto front json file
                 ps_file_path=get_path("asr_ps.json"),  # path to NASBenchGraph Pareto set json file
                 normalized_objectives=True,  # whether to normalize the objectives
                 ):
        search_space = NASBenchGraphSearchSpace()
        evaluator = NASBenchGraphEvaluator(objs=objs)
        super().__init__(search_space, evaluator, normalized_objectives)

        self.pf = np.array(json.load(open(pf_file_path, 'r')).get(objs, [1]))
        self.ps = np.array(json.load(open(ps_file_path, 'r')).get(objs, [1]))

    @property
    def name(self):
        return 'NASBenchGraphBenchmark'

    @property
    def pareto_front(self):
        return self.pf

    @property
    def pareto_set(self):
        return self.ps

    def debug(self):
        archs = self.search_space.sample(10)
        X = self.search_space.encode(archs)
        F = self.evaluate(X, true_eval=True)
        igd = self.calc_perf_indicator(X, 'igd')
        hv = self.calc_perf_indicator(X, 'hv')
        norm_hv = self.calc_perf_indicator(X, 'normalized_hv')

        ps_igd = self.calc_perf_indicator(self.pareto_set, 'igd')
        ps_hv = self.calc_perf_indicator(self.pareto_set, 'hv')
        ps_norm_hv = self.calc_perf_indicator(self.pareto_set, 'normalized_hv')

        print(archs)
        print(X)
        print(F)
        print(igd)
        print(hv)
        print(norm_hv)

        print("PF IGD: {}, this number should be really close to 0".format(ps_igd))
        print(ps_hv)
        print("PF normalized HV: {}, this number should be really close to 1".format(ps_norm_hv))


if __name__ == '__main__':
    # for i in NASBenchGraphResult.objects.all():
    #     if i.genotype.find('(') < 0:
    #         i.genotype = i.genotype.replace('[', '(').replace(']', ')')
    #         i.genotype = '[' + i.genotype[1:-1] + ']'
    #         i.save()
    #         print(i.genotype)
    # archs = [i.genotype for i in NASBenchGraphResult.objects.all()]
    # archs.sort()
    # ans = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    # print(archs[-10:])
    # for i in archs:
    #     I = eval(i.replace('(', '').replace(')', ''))
    #     # print(I)
    #     for index, j in enumerate(I):
    #         ans[index] = max(ans[index], j)
    # print(ans)
    pf = json.load(open('/home/satan/Desktop/evoxbench-dev-main/data/nbasr/asr_pf.json', 'r'))
    ps = json.load(open('/home/satan/Desktop/evoxbench-dev-main/data/nbasr/asr_ps.json', 'r'))
    objs = 'err&params'
    eva = NASBenchGraphEvaluator(objs=objs)
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    archs = [i.genotype for i in NASBenchGraphResult.objects.all()]
    F = eva.evaluate(archs, true_eval=True)
    F = np.array([list(v.values()) for v in F])
    
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    print(front.shape)
    print(front[:10])
    pf[objs] = F[front].tolist()
    ps[objs] = [archs[i] for i in front]
    
    json.dump(pf, open('/home/satan/Desktop/evoxbench-dev-main/data/nbasr/asr_pf.json', 'w'))
    json.dump(ps, open('/home/satan/Desktop/evoxbench-dev-main/data/nbasr/asr_ps.json', 'w'))

    benchmark = NASBenchGraphBenchmark(
        objs='err&params&flops',
        normalized_objectives=False
    )

    benchmark.debug()

