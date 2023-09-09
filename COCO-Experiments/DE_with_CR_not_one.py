#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A short and simple example experiment with restarts.
The script is fully functional but also emphasises on readability. It
features restarts, timings and recording termination conditions.
To benchmark a different solver, `fmin` must be re-assigned and another
`elif` block added around line 119 to account for the solver-specific
call.
When calling the script, previously assigned variables can be re-assigned
via a ``name=value`` argument without white spaces, where ``value`` is
interpreted as a single python literal. Additionally, ``batch`` is recognized
as argument defining the `current_batch` number and the number of `batches`,
like ``batch=2/8`` runs batch 2 of 8.
Examples, preceeded by "python" in an OS shell and by "run" in an IPython
shell::
    example_experiment2.py budget_multiplier=3  # times dimension
    example_experiment2.py budget_multiplier=1e4 cocopp=None  # omit post-processing
    
    example_experiment2.py budget_multiplier=1e4 suite_name=bbob-biobj
    example_experiment2.py budget_multiplier=1000 batch=1/16
Post-processing with `cocopp` is only invoked in the single-batch case.
Details: ``batch=9/8`` is equivalent to ``batch=1/8``. The first number
is taken modulo to the second.
See the code: `<https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment2.py>`_
See a beginners example experiment: `<https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment_for_beginners.py>`_
"""
from __future__ import division, print_function, unicode_literals
__author__ = "Nikolaus Hansen and ..."
import sys
import time  # output some timings per evaluation
from collections import defaultdict
import os, webbrowser  # to show post-processed results in the browser
import numpy as np  # for median, zeros, random, asarray
import cocoex  # experimentation module
import cma
try: import cocopp  # post-processing module
except: pass
import portion as p
import random
import matplotlib.pyplot as plt

### MKL bug fix
def set_num_threads(nt=1, disp=1):
    """see https://github.com/numbbo/coco/issues/1919
    and https://twitter.com/jeremyphoward/status/1185044752753815552
    """
    try: import mkl
    except ImportError: disp and print("mkl is not installed")
    else:
        mkl.set_num_threads(nt)
    nt = str(nt)
    for name in ['OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS',
                 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS']:
        os.environ[name] = nt
    disp and print("setting mkl threads num to", nt)

if sys.platform.lower() not in ('darwin', 'windows'):
    set_num_threads(1)

### solver imports (add other imports if necessary)
import scipy.optimize  # to define the solver to be benchmarked
try: import cma
except: pass  # may not be installed

def random_search(f, lbounds, ubounds, evals):
    """Won't work (well or at all) for `evals` much larger than 1e5"""
    [f(x) for x in np.asarray(lbounds) + (np.asarray(ubounds) - lbounds)
                               * np.random.rand(int(evals), len(ubounds))]


def DifferentialEvolution_CR_1(objective, dimension:int, popsize:int, crossover:float, diffweight:float, T:int, seed = None):
    # constraints on the parameters
    dimensionvalid = list(range(1, 50))
    if dimension not in dimensionvalid:
        raise ValueError('DifferentialEvolution: dimension should be between 1 and 50')
    crossovervalid = p.closed(0,1)
    if crossover not in crossovervalid:
        raise ValueError('DifferentialEvolution: crossover should be a probability')
    diffweightvalid = p.closed(0, 2)
    if diffweight not in diffweightvalid:
        raise ValueError('DifferentialEvolution: diffweight should be in %r' % diffweightvalid)
    if popsize < 4:
        raise ValueError('DifferentialEvolution: popsize should be >= 4')
    '''
    we are considering function whose domain is R^d
    unconstrained optimization.
    '''
    # Initial vectors
    if seed is not None:
        np.random.seed(seed)
    vectors = [np.random.random(dimension) for _ in range(popsize)]
    #  >>>>>>> Begin Algorithm <<<<<<<<<<<<
    for i in range(T):
        vectors = np.array(vectors)  # make sure it is an np.array 
        # picking three different vectors for each vector
        for index, vector in enumerate(vectors):
            list_no_vector = list(np.delete(vectors, index, axis = 0))
            randomvectors = random.sample(list_no_vector, 3)  # For NumPy arrays
             # OR
             # randomvectors = random.sample([ele for ele in vectors if ele != vector], 3)  # For lists
            R = random.randint(1, dimension)
            y = np.zeros(dimension)
            # update dimensions
            for j in range(dimension):
                r = np.random.uniform(0,1)
                if  r < crossover or j == R:
                    y[j] = (randomvectors[0][j] + diffweight * (randomvectors[1][j] - randomvectors[2][j]))
                else: 
                    y[j] = (vector[j])
            if objective(y) < objective(vector):
                vectors[index] = y
    # >>>>>>>>>>  End Algorithm  <<<<<<<<<<<<



### input (to be modified if necessary/desired)
# fmin = scipy.optimize.fmin
#fmin = scipy.optimize.fmin_slsqp
# fmin = scipy.optimize.fmin_cobyla
#fmin = cocoex.solvers.random_search
fmin = DifferentialEvolution_CR_1

suite_name = "bbob"  # see cocoex.known_suite_names
budget_multiplier = 1e5  # times dimension, increase to 10, 100, ...
suite_filter_options = (""  # without filtering, a suite has instance_indices 1-15
                        # "dimensions: 2,3,5,10,20 "  # skip dimension 40
                        # "instance_indices: 1-5 "  # relative to suite instances
                       )
# for more suite filter options see http://numbbo.github.io/coco-doc/C/#suite-parameters
suite_year_option = ""  # "year: 2022"  # determine instances by year, not all years work for all suites :-(

batches = 1  # number of batches, batch=3/32 works to set both, current_batch and batches
current_batch = 1  # only current_batch modulo batches is relevant
output_folder = ''

### possibly modify/overwrite above input parameters from input args
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', 'help', '-help', '--help'):
        print(__doc__)
        raise ValueError("printed help and aborted")
    input_params = cocoex.utilities.args_to_dict(
        sys.argv[1:], globals(), {'batch': 'current_batch/batches'}, print=print)
    globals().update(input_params)  # (re-)assign variables

# extend output folder input parameter, comment out if desired otherwise
output_folder += '%s_of_%s_%dD_on_%s' % (
        fmin.__name__, fmin.__module__, int(budget_multiplier), suite_name)

if batches > 1:
    output_folder += "_batch%03dof%d" % (current_batch, batches)

### prepare
suite = cocoex.Suite(suite_name, suite_year_option, suite_filter_options)
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()
stoppings = defaultdict(list)  # dict of lists, key is the problem index
timings = defaultdict(list)  # key is the dimension

### go
print('*** benchmarking %s from %s on suite %s ***'
      % (fmin.__name__, fmin.__module__, suite_name))
time0 = time.time()
for batch_counter, problem in enumerate(suite):  # this loop may take hours or days...
    if batch_counter % batches != current_batch % batches:
        continue
    if not len(timings[problem.dimension]) and len(timings) > 1:
        print("\n   %s %d-D done in %.1e seconds/evaluations"
              % (minimal_print.stime, sorted(timings)[-2],
                 np.median(timings[sorted(timings)[-2]])), end='')
    problem.observe_with(observer)  # generate the data for cocopp post-processing
    problem(np.zeros(problem.dimension))  # making algorithms more comparable
    propose_x0 = problem.initial_solution_proposal  # callable, all zeros in first call
    evalsleft = lambda: int(problem.dimension * budget_multiplier + 1 -
                            max((problem.evaluations, problem.evaluations_constraints)))
    time1 = time.time()
    # apply restarts
    irestart = -1
    while evalsleft() > 0 and not problem.final_target_hit:
        irestart += 1

        # here we assume that `fmin` evaluates the final/returned solution
        if 11 < 3:  # add solver to investigate here
            pass
        elif fmin is scipy.optimize.fmin:
            output = fmin(problem, propose_x0(), maxfun=evalsleft(), disp=False, full_output=True)
            stoppings[problem.index].append(output[4])
        elif fmin is scipy.optimize.fmin_slsqp:
            output = fmin(problem, propose_x0(), iter=int(evalsleft() / problem.dimension + 1),  # very approximate way to respect budget
                          full_output=True, iprint = -1)
            # print(problem.dimension, problem.evaluations)
            stoppings[problem.index].append(output[3:])
        elif fmin in (cocoex.solvers.random_search, random_search):
            fmin(problem, problem.lower_bounds, problem.upper_bounds, evalsleft())
        elif fmin.__name__ == 'fmin2' and 'cma' in fmin.__module__:  # cma.fmin2:
            xopt, es = fmin(problem, propose_x0, 2,
                            {'maxfevals':evalsleft(), 'verbose':-9, 'CMA_rankone': 0}, restarts=30)
            stoppings[problem.index].append(es.stop())
        elif fmin is scipy.optimize.fmin_cobyla:
            fmin(problem, propose_x0(), lambda x: -problem.constraint(x), maxfun=evalsleft(),
                 disp=0, rhoend=1e-9)
        elif fmin is DifferentialEvolution_CR_1:
            fmin(problem, problem.dimension, (10*problem.dimension), 0.4, 0.8, int(evalsleft() / 2*10*problem.dimension))

    timings[problem.dimension].append((time.time() - time1) / problem.evaluations
                                      if problem.evaluations else 0)
    minimal_print(problem, restarted=irestart, final=problem.index == len(suite) - 1)
    with open(output_folder + '_stopping_conditions.pydict', 'wt') as file_:
        file_.write("# code to read in these data:\n"
                    "# import ast\n"
                    "# with open('%s_stopping_conditions.pydict', 'rt') as file_:\n"
                    "#     stoppings = ast.literal_eval(file_.read())\n"
                    % output_folder)
        file_.write(repr(dict(stoppings)))

### print timings and final message
print("\n   %s %d-D done in %.1e seconds/evaluations"
      % (minimal_print.stime, sorted(timings)[-1], np.median(timings[sorted(timings)[-1]])))
if batches > 1:
    print("*** Batch %d of %d batches finished in %s."
          " Make sure to run *all* batches (via current_batch or batch=#/#) ***"
          % (current_batch, batches, cocoex.utilities.ascetime(time.time() - time0)))
else:
    print("*** Full experiment done in %s ***"
          % cocoex.utilities.ascetime(time.time() - time0))

print("Timing summary:\n"
      "  dimension  median seconds/evaluations\n"
      "  -------------------------------------")
for dimension in sorted(timings):
    print("    %3d       %.1e" % (dimension, np.median(timings[dimension])))
print("  -------------------------------------")

### post-process data
if batches == 1 and 'cocopp' in globals() and cocopp not in (None, 'None'):
    cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
    webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")