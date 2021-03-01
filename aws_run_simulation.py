import time
import pickle
import sys, os

from superscript_model import model

## TODO: ADD save project level tracked data (see scatter_plot.py)

STEPS = 100

MODEL_DIR = os.path.realpath(os.path.dirname('./superscript_model'))
sys.path.append(os.path.normpath(MODEL_DIR))

abm = model.SuperScriptModel(1000)

start_time = time.time()
abm.run_model(STEPS)
elapsed_time = time.time() - start_time
print("Took %.2f seconds to run %d steps." % (elapsed_time, STEPS))

tracked = abm.datacollector.get_model_vars_dataframe()
print(tracked)

with open('tracked_test_1000.pickle', 'wb') as ofile:
    pickle.dump(tracked, ofile)
