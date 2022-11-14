""" NASBench 101 Benchmark """
import copy
import json
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
from nasbenchmacro.models import NASBenchMacroResult  # has to be imported after the init method

__all__ = ['NASBenchMacroSearchSpace', 'NASBenchMacroEvaluator', 'NASBenchMacroBenchmark']

HASH = {'conv3x3-bn-relu': 0, 'conv1x1-bn-relu': 1, 'maxpool3x3': 2}


def get_path(name):
    print(name)
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "nbmacro" / name)


class NASBenchMacroSearchSpace(SearchSpace):
    """
        NASBenchMacro API need to be first installed following the official instructions from
        https://github.com/google-research/nasbench
    """

    def __init__(self, **kwargs
                 ):
        super().__init__(**kwargs)

        self.n_var = 8
        self.lb = [0] * self.n_var
        self.ub = [2] * self.n_var

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    @property
    def name(self):
        return 'NASBenchMacroSearchSpace'

    def _sample(self, phenotype=True):
        tmp = [random.randint(self.lb[i], self.ub[i]) for i in range(len(self.lb))]
        arch = ''.join(map(str, tmp))
        return arch if phenotype else self._encode(arch)

    def _encode(self, arch: str):
        # encode architecture phenotype to genotype
        # a sample architecture
        # '64:56:8:24:16:24:48:8'
        ans = np.empty(self.n_var, dtype=np.int64)
        for index, i in enumerate(arch):
            ans[index] = int(i)
        return ans

    def _decode(self, x: ndarray) -> str:
        return ''.join(map(str, x.tolist()))

    def visualize(self, arch):
        raise NotImplementedError


class NASBenchMacroEvaluator(Evaluator):
    def __init__(self,
                 objs='err&params&flops',  # objectives to be minimized
                 ):
        super().__init__(objs)


    @property
    def name(self):
        return 'NASBenchMacroEvaluator'

    def evaluate(self, archs, objs=None,
                 true_eval=False  # query the true (mean over three runs) performance
                 ):

        if objs is None:
            objs = self.objs

        batch_stats = []
        results = {i.genotype: i for i in NASBenchMacroResult.objects.filter(genotype__in=archs)}
        for i, arch in enumerate(archs):
            stats = {}
            ans = results[arch]
            if true_eval:
                top1 = np.mean(ans.test_acc['acc'])
            else:
                top1 = np.random.choice(ans.test_acc['acc'])
            params = ans.params
            flops = ans.flops
            if 'err' in objs:
                stats['err'] = 100 - top1
            if 'params' in objs:
                stats['params'] = params
            if 'flops' in objs:
                stats['flops'] = flops
            batch_stats.append(stats)

        return batch_stats


class NASBenchMacroBenchmark(Benchmark):
    def __init__(self,

                 objs='err&params&flops',  # objectives to be minimized
                 pf_file_path=get_path("macro_pf.json"),  # path to NASBenchMacro Pareto front json file
                 ps_file_path=get_path("macro_ps.json"),  # path to NASBenchMacro Pareto set json file
                 normalized_objectives=True,  # whether to normalize the objectives
                 ):
        search_space = NASBenchMacroSearchSpace()
        evaluator = NASBenchMacroEvaluator(objs=objs)
        super().__init__(search_space, evaluator, normalized_objectives)

        self.pf = np.array(json.load(open(pf_file_path, 'r')).get(objs, [1]))
        self.ps = np.array(json.load(open(ps_file_path, 'r')).get(objs, [1]))

    @property
    def name(self):
        return 'NASBenchMacroBenchmark'

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
    # pf = json.load(open('/home/satan/Desktop/evoxbench-dev-main/data/nbmacro/macro_pf.json', 'r'))
    # ps = json.load(open('/home/satan/Desktop/evoxbench-dev-main/data/nbmacro/macro_ps.json', 'r'))
    # objs = 'err&flops'
    # eva = NASBenchMacroEvaluator(objs=objs)
    # from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    # archs = [i.genotype for i in NASBenchMacroResult.objects.all()]
    # F = eva.evaluate(archs, true_eval=True)
    # F = np.array([list(v.values()) for v in F])
    #
    # front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    # print(front.shape)
    # print(front[:10])
    # pf[objs] = F[front].tolist()
    # ps[objs] = [archs[i] for i in front]
    #
    # json.dump(pf, open('/home/satan/Desktop/evoxbench-dev-main/data/nbmacro/macro_pf.json', 'w'))
    # json.dump(ps, open('/home/satan/Desktop/evoxbench-dev-main/data/nbmacro/macro_ps.json', 'w'))

    benchmark = NASBenchMacroBenchmark(
        objs='err&params&flops',
        normalized_objectives=False
    )

    benchmark.debug()

