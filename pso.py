import os
import argparse

import htm.optimization.ae as auto_exp
import htm.optimization.optimizers as optimizers
from htm.optimization.swarming import ParticleSwarmOptimization

all_optimizers = [
    optimizers.EvaluateDefaultParameters,
    optimizers.EvaluateAllExperiments,
    optimizers.EvaluateBestExperiment,
    optimizers.EvaluateHashes,
    optimizers.GridSearch,
    optimizers.CombineBest,
    ParticleSwarmOptimization,
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Passed onto the experiment\'s main function.')
    parser.add_argument('--tag', type=str,
        help='Optional string appended to the name of the AE directory.  Use tags to '
             'keep multiple variants of an experiment alive and working at the same time.')
    parser.add_argument('-n', '--processes',  type=int, default=os.cpu_count(),
        help='Number of experiments to run simultaneously, defaults to the number of CPU cores available.')
    parser.add_argument('--time_limit',  type=float, default=None,
        help='Hours, time limit for each run of the experiment.',)
    parser.add_argument('--memory_limit',  type=float, default=None,
        help='Gigabytes, RAM memory limit for each run of the experiment.')
    parser.add_argument('--parse',  action='store_true',
        help='Parse the lab report and write it back to the same file, then exit.')
    parser.add_argument('--rmz', action='store_true',
        help='Remove all experiments which have zero attempts.')
    parser.add_argument('experiment', nargs=argparse.REMAINDER,
        help='Name of experiment module followed by its command line arguments.')

    assert( all( issubclass(Z, optimizers.BaseOptimizer) for Z in all_optimizers))
    for method in all_optimizers:
        method.add_arguments(parser)

    args = parser.parse_args()
    selected_method = [X for X in all_optimizers if X.use_this_optimizer(args)]

    ae = auto_exp.Laboratory(args.experiment,
        tag      = args.tag,
        verbose  = args.verbose)
    ae.save()
    print("Lab Report written to %s"%ae.lab_report)

    if args.parse:
        pass

    elif args.rmz:
        for x in ae.experiments:
            if x.attempts == 0:
                ae.experiments.remove(x)
                ae.experiment_ids.pop(hash(x))
        ae.save()
        print("Removed all experiments which had not yet been attempted.")

    elif not selected_method:
        print("Error: missing argument for what to to.")
    elif len(selected_method) > 1:
        print("Error: too many argument for what to to.")
    else:
        ae.method = selected_method[0]( ae, args )

        giga = 2**30
        if args.memory_limit is not None:
            memory_limit = int(args.memory_limit * giga)
        else:
            # TODO: Not X-Platform, replace with "psutil.virtual_memory.available"
            available_memory = int(os.popen("free -b").readlines()[1].split()[3])
            memory_limit = int(available_memory / args.processes)
            print("Memory Limit %.2g GB per instance."%(memory_limit / giga))

        ae.run( processes    = args.processes,
                time_limit   = args.time_limit,
                memory_limit = memory_limit,)

    print("Exit.")