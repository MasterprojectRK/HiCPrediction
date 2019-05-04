
import logging
import numpy as np
import sm1 as sm
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

def callTraining(cfg):
    args = ['-a', 'train']
    args = sm.parseArguments(args)
    cfg = {k : cfg[k] for k in cfg}
    args.learningRate = cfg["lr"]
    args.hidden = cfg["hidden"]
    args.epochs = 2000
    args.outputSize = 1000
    args.output= cfg["output"]
    args.convSize= cfg["lr"]
    args.convSize= cfg["cs"]
    args.firstLayer= cfg["fl"]
    args.secondLayer = cfg["sl"]
    args.thirdLayer = cfg["tl"]
    args.lastLayer = cfg["ll"]
    args.dropout1 = cfg["d1"]
    args.dropout2 = cfg["d2"]
    args.pool1 = cfg["p1"]
    args.loss = cfg["loss"]
    args.prepData = cfg["prepData"]
    args.saveModel = False
    args.dimAfterP1 = int(args.cutWidth / args.pool1)
    args.dimAfterP2 = int(args.dimAfterP1 / args.pool1)
    args.padding = int(np.floor(args.convSize / 2) )
    print(args)
    return sm.startTraining(args)

logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output
cs = ConfigurationSpace()
hidden = CategoricalHyperparameter("hidden", ["S", "T", "TR", "R"],default_value="S")
output = CategoricalHyperparameter("output", ["S", "T", "TR", "R"],default_value="S")
lr = CategoricalHyperparameter("lr", [0.0005, 0.001,0.0001,0.00005],default_value=0.001)
cos = CategoricalHyperparameter("cs", [3,5,7,9],default_value=5)
fl = CategoricalHyperparameter("fl", [2,3,4,6,8,10],default_value=4)
sl = CategoricalHyperparameter("sl", [2,3,4,6,8,10],default_value=4)
tl = CategoricalHyperparameter("tl", [2,3,4,6, 8, 10],default_value=4)
d1 = CategoricalHyperparameter("d1", [0,0.1,0.2,0.3,0.4,0.5],default_value=0.1)
d2 = CategoricalHyperparameter("d2", [0,0.1,0.2,0.3,0.4,0.5],default_value=0.1)
p1 = CategoricalHyperparameter("p1", [1,2],default_value=2)
loss = CategoricalHyperparameter("loss", ["L1"],default_value="L1")
ll = CategoricalHyperparameter("ll",["first","second",
                                            "third"],default_value="third")
pd = CategoricalHyperparameter("prepData",["customLog"],default_value="customLog")
cs.add_hyperparameters([lr, hidden, output, cos, fl, sl, tl,ll, d1,d2,p1,loss, pd])

# Others are kernel-specific, so we can add conditions to limit the searchspace
# degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)     # Only used by kernel poly
# coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
# cs.add_hyperparameters([degree, coef0])
# use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
# use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
# cs.add_conditions([use_degree, use_coef0])

# This also works for parameters that are a mix of categorical and values from a range of numbers
# For example, gamma can be either "auto" or a fixed float
# gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
# gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
# cs.add_hyperparameters([gamma, gamma_value])
# We only activate gamma_value if gamma is set to "value"
# cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
# And again we can restrict the use of gamma in general to the choice of the kernel
# cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))


# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 200,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "false",
                       "shared_model": True,
                "input_psmac_dirs": "smac3-output*"
                     })

# def_value = callTraining(cs.get_default_configuration())
# print("Default Value: %.6f" % (def_value))

print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC(scenario=scenario, rng=np.random.RandomState(9),
        tae_runner=callTraining)

incumbent = smac.optimize()

inc_value = callTraining(incumbent)

print("Optimized Value: %.6f" % (inc_value))


