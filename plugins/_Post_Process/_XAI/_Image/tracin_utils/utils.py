# Copyright 2023 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import csv
import json
import yaml
import numpy as np
import nnabla as nn
from nnabla.logger import logger
from nnabla.ext_utils import get_extension_context
from nnabla.utils.load import load
from shutil import rmtree


class DictInterfaceFactory(object):
    '''Creating a single dict interface of any function or class.

    Example:

    .. code-block:: python
        # Define a function.
        def foo(a, b=1, c=None):
            for k, v in locals():
                print(k, v)

        # Register the function to the factory.
        dictif = DictInterfaceFactory()
        dictif.register(foo)

        # You can call the registered function by name and a dict representing the arguments.
        cfg = dict(a=1, c='hello')
        dictif.call('foo', cfg)

        # The following will fail because the `foo` function requires `a`.
        #     cfg = dict(c='hello')
        #     dictif.call('foo', cfg)

        # Any argument not required will be just ignored.
        cfg = dict(a=1, aaa=0)
        dictif.call('foo', cfg)

        # You can also use it for class initializer (we use it as a class decorator).
        @dictif.register
        class Bar:
            def __init__(self, a, b, c=None):
                for k, v in locals():
                    print(k, v)

        bar = dictif.call('Bar', dict(a=0, b=0))

    '''

    def __init__(self):
        self._factory = {}

    def register(self, cls):
        import inspect

        # config interface function
        def func(cfg):
            sig = inspect.signature(cls)
            # Handle all arguments of the created class
            args = {}
            for p in sig.parameters.values():
                # Positional argument
                if p.default is p.empty and p.name not in cfg:
                    raise ValueError(
                        f'`{cls.__name__}`` requires an argument `{p.name}`. Not found in cfg={cfg}.')
                args[p.name] = cfg.get(p.name, p.default)
            return cls(**args)

        # Register config interface function
        self._factory[cls.__name__] = func
        return cls

    def call(self, name, cfg):
        if name in self._factory:
            return self._factory[name](cfg)
        raise ValueError(
            f'`{name}`` not found in `{list(self._factory.keys())}`.')


# Creater function
_lr_sched_factory = DictInterfaceFactory()


def create_learning_rate_scheduler(cfg):
    '''
    Create a learning rate scheduler from config.

    Args:
        cfg (dict-like object):
            It must contain ``scheduler_type`` to specify a learning rate scheduler class.

    Returns:
        Learning rate scheduler object.

    Example:

    With the following yaml file (``example.yaml``),

    .. code-block:: yaml

        learning_rate_scheduler:
            scheduler_type: EpochStepLearningRateScheduler
            base_lr: 1e-2
            decay_at: [40, 65]
            decay_rate: 0.1
            power: 1  # Ignored in EpochStepLearningRateScheduler

    you can create a learning rate scheduler class as following.

    .. code-block:: python

        from neu.yaml_wrapper import read_yaml
        cfg = read_yaml('example.yaml)

        lr_sched = create_learning_rate_scheduler(cfg.learning_rate_scheduler)

    '''

    return _lr_sched_factory.call(cfg.scheduler_type, cfg)


class BaseLearningRateScheduler(object):
    '''
    Base class of Learning rate scheduler.

    This gives a current learning rate according to a scheduling logic
    implemented as a method `_get_lr` in a derived class. It internally
    holds the current epoch and the current iteration to calculate a
    scheduled learning rate. You can get the current learning rate by
    calling `get_lr`. You have to set the current epoch which will be
    used in `_get_lr` by manually calling  `set_epoch(self, epoch)`
    while it updates the current iteration when you call
    `get_lr_and_update`.

    Example:

    .. code-block:: python
        class EpochStepLearningRateScheduler(BaseLearningRateScheduler):
            def __init__(self, base_lr, decay_at=[30, 60, 80], decay_rate=0.1, warmup_epochs=5):
                self.base_learning_rate = base_lr
                self.decay_at = np.asarray(decay_at, dtype=np.int32)
                self.decay_rate = decay_rate
                self.warmup_epochs = warmup_epochs

            def _get_lr(self, current_epoch, current_iter):
                # This scheduler warmups and decays using current_epoch
                # instead of current_iter
                lr = self.base_learning_rate
                if current_epoch < self.warmup_epochs:
                    lr = lr * (current_epoch + 1) / (self.warmup_epochs + 1)
                    return lr

                p = np.sum(self.decay_at <= current_epoch)
                return lr * (self.decay_rate ** p)

        def train(...):

            ...

            solver = Momentum()
            lr_sched = EpochStepLearningRateScheduler(1e-1)
            for epoch in range(max_epoch):
                lr_sched.set_epoch(epoch)
                for it in range(max_iter_in_epoch):
                    lr = lr_sched.get_lr_and_update()
                    solver.set_learning_rate(lr)
                    ...


    '''

    def __init__(self):
        self._iter = 0
        self._epoch = 0

    def set_iter_per_epoch(self, it):
        pass

    def set_epoch(self, epoch):
        '''Set current epoch number.
        '''
        self._epoch = epoch

    def get_lr_and_update(self):
        '''
        Get current learning rate and update itereation count.

        The iteration count is calculated by how many times this method is called.

        Returns: Current learning rate

        '''
        lr = self.get_lr()
        self._iter += 1
        return lr

    def get_lr(self):
        '''
        Get current learning rate according to the schedule.
        '''
        return self._get_lr(self._epoch, self._iter)

    def _get_lr(self, current_epoch, current_iter):
        '''
        Get learning rate by current iteration.

        Args:
            current_epoch(int): Epoch count.
            current_iter(int):
                Current iteration count from the beginning of training.

        Note:
            A derived class must override this method. 

        '''
        raise NotImplementedError('')


@_lr_sched_factory.register
class EpochStepLearningRateScheduler(BaseLearningRateScheduler):
    '''
    Learning rate scheduler with step decay.

    Args:
        base_lr (float): Base learning rate
        decay_at (list of ints): It decays the lr by a factor of `decay_rate`.
        decay_rate (float): See above.
        warmup_epochs (int): It performs warmup during this period.
        legacy_warmup (bool):
            We add 1 in the denominator to be consistent with previous
            implementation.

    '''

    def __init__(self, base_lr, decay_at=[30, 60, 80], decay_rate=0.1, warmup_epochs=5, legacy_warmup=False):
        super().__init__()
        self.base_learning_rate = base_lr
        self.decay_at = np.asarray(decay_at, dtype=np.int32)
        self.decay_rate = decay_rate
        self.warmup_epochs = warmup_epochs
        self.legacy_warmup_denom = 1 if legacy_warmup else 0

    def _get_lr(self, current_epoch, current_iter):
        lr = self.base_learning_rate
        # Warmup
        if current_epoch < self.warmup_epochs:
            lr = lr * (current_epoch + 1) \
                / (self.warmup_epochs + self.legacy_warmup_denom)
            return lr

        p = np.sum(self.decay_at <= current_epoch)
        return lr * (self.decay_rate ** p)


class AttrDict(dict):
    # special internal variable used for error message.
    _parent = []

    def __setattr__(self, key, value):
        if key == "_parent":
            self.__dict__["_parent"] = value
            return

        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(
                "dict (AttrDict) has no chain of attributes '{}'".format(".".join(self._parent + [key])))

        if isinstance(self[key], dict):
            self[key] = AttrDict(self[key])
            self[key]._parent = self._parent + [key]

        return self[key]

    def dump_to_stdout(self):
        print("================================configs================================")
        for k, v in self.items():
            print("{}: {}".format(k, v))

        print("=======================================================================")


def read_yaml(filepath):
    with open(filepath, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return AttrDict(data)


def save_nnp(input, output, batchsize):
    runtime_contents = {
        'networks': [
            {'name': 'Validation',
             'batch_size': batchsize,
             'outputs': output,
             'names': input}],
        'executors': [
            {'name': 'Runtime',
             'network': 'Validation',
             'data': [k for k, _ in input.items()],
             'output': [k for k, _ in output.items()]}]}
    return runtime_contents


def save_to_csv(filename, header, list_to_save, data_type):
    with open(filename, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(np.array([tuple(row)
                                   for row in list_to_save], dtype=data_type))


def save_checkpoint(path, current_iter, solvers):
    """Saves the checkpoint file which contains the params and its state info.

        Args:
            path: Path to the directory the checkpoint file is stored in.
            current_iter: Current iteretion of the training loop.
            solvers: A dictionary about solver's info, which is like;
                     solvers = {"identifier_for_solver_0": solver_0,
                               {"identifier_for_solver_1": solver_1, ...}
                     The keys are used just for state's filenames, so can be anything.
                     Also, you can give a solver object if only one solver exists.
                     Then, the "" is used as an identifier.

        Examples:
            # Create computation graph with parameters.
            pred = construct_pred_net(input_Variable, ...)

            # Create solver and set parameters.
            solver = S.Adam(learning_rate)
            solver.set_parameters(nn.get_parameters())

            # If you have another_solver like,
            # another_solver = S.Sgd(learning_rate)
            # another_solver.set_parameters(nn.get_parameters())

            # Training loop.
            for i in range(start_point, max_iter):
                pred.forward()
                pred.backward()
                solver.zero_grad()
                solver.update()
                save_checkpoint(path, i, solver)

                # If you have another_solver,
                # save_checkpoint(path, i,
                      {"solver": solver, "another_solver": another})

        Notes:
            It generates the checkpoint file (.json) which is like;
            checkpoint_1000 = {
                    "":{
                        "states_path": <path to the states file>
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

            If you have multiple solvers.
            checkpoint_1000 = {
                    "generator":{
                        "states_path": <path to the states file>,
                        "params_names":["deconv1/conv/W", ...],
                        "num_update":1000
                       },
                    "discriminator":{
                        "states_path": <path to the states file>,
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

    """

    if isinstance(solvers, nn.solver.Solver):
        solvers = {"": solvers}

    checkpoint_info = dict()

    for solvername, solver_obj in solvers.items():
        prefix = "{}_".format(solvername.replace(
            "/", "_")) if solvername else ""
        partial_info = dict()

        # save solver states.
        states_fname = prefix + 'states_{}.h5'.format(current_iter)
        states_path = os.path.join(path, states_fname)
        solver_obj.save_states(states_path)
        partial_info["states_path"] = states_path

        # save registered parameters' name. (just in case)
        params_names = [k for k in solver_obj.get_parameters().keys()]
        partial_info["params_names"] = params_names

        # save the number of solver update.
        num_update = getattr(solver_obj.get_states()[params_names[0]], "t")
        partial_info["num_update"] = num_update

        checkpoint_info[solvername] = partial_info

    # save parameters.
    params_fname = 'params_{}.h5'.format(current_iter)
    params_path = os.path.join(path, params_fname)
    nn.parameter.save_parameters(params_path)
    checkpoint_info["params_path"] = params_path
    checkpoint_info["current_iter"] = current_iter

    checkpoint_fname = 'checkpoint_{}.json'.format(current_iter)
    filename = os.path.join(path, checkpoint_fname)

    with open(filename, 'w') as f:
        json.dump(checkpoint_info, f)

    logger.info("Checkpoint save (.json): {}".format(filename))

    return


def load_checkpoint(path, solvers):
    """Given the checkpoint file, loads the parameters and solver states.

        Args:
            path: Path to the checkpoint file.
            solvers: A dictionary about solver's info, which is like;
                     solvers = {"identifier_for_solver_0": solver_0,
                               {"identifier_for_solver_1": solver_1, ...}
                     The keys are used for retrieving proper info from the checkpoint.
                     so must be the same as the one used when saved.
                     Also, you can give a solver object if only one solver exists.
                     Then, the "" is used as an identifier.

        Returns:
            current_iter: The number of iteretions that the training resumes from.
                          Note that this assumes that the numbers of the update for
                          each solvers is the same.

        Examples:
            # Create computation graph with parameters.
            pred = construct_pred_net(input_Variable, ...)

            # Create solver and set parameters.
            solver = S.Adam(learning_rate)
            solver.set_parameters(nn.get_parameters())

            # AFTER setting parameters.
            start_point = load_checkpoint(path, solver)

            # Training loop.

        Notes:
            It requires the checkpoint file. For details, refer to save_checkpoint;
            checkpoint_1000 = {
                    "":{
                        "states_path": <path to the states file>
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

            If you have multiple solvers.
            checkpoint_1000 = {
                    "generator":{
                        "states_path": <path to the states file>,
                        "params_names":["deconv1/conv/W", ...],
                        "num_update":1000
                       },
                    "discriminator":{
                        "states_path": <path to the states file>,
                        "params_names":["conv1/conv/W", ...],
                        "num_update":1000
                       },
                    "current_iter": 1000
                    }

    """

    assert os.path.isfile(path), "checkpoint file not found"

    with open(path, 'r') as f:
        checkpoint_info = json.load(f)

    if isinstance(solvers, nn.solver.Solver):
        solvers = {"": solvers}

    logger.info("Checkpoint load (.json): {}".format(path))

    # load parameters (stored in global).
    params_path = checkpoint_info["params_path"]
    assert os.path.isfile(params_path), "parameters file not found."

    nn.parameter.load_parameters(params_path)

    for solvername, solver_obj in solvers.items():
        partial_info = checkpoint_info[solvername]
        if set(solver_obj.get_parameters().keys()) != set(partial_info["params_names"]):
            logger.warning("Detected parameters do not match.")

        # load solver states.
        states_path = partial_info["states_path"]
        assert os.path.isfile(states_path), "states file not found."

        # set solver states.
        solver_obj.load_states(states_path)

    # get current iteration. note that this might differ from the numbers of
    # update.
    current_iter = checkpoint_info["current_iter"]

    return current_iter


def get_context(device_id):
    # for cli app use
    try:
        context = 'cudnn'
        ctx = get_extension_context(context, device_id=device_id)
    except (ModuleNotFoundError, ImportError):
        context = 'cpu'
        ctx = get_extension_context(context, device_id=device_id)
    # for nnc use
    config_filename = 'net.nntxt'
    if os.path.isfile(config_filename):
        config_info = load([config_filename])
        ctx = config_info.global_config.default_context

    return ctx


def delete_dir(dir_name, keyword='tracin_infl_results'):
    if os.path.isdir(dir_name):
        if keyword in dir_name:
            rmtree(dir_name)


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
