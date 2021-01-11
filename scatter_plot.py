from superscript_model.model import SuperScriptModel

import matplotlib.pyplot as plt

abm = SuperScriptModel(100)
abm.run_model(10)

# We are going to use the DataCollector to collect information on project from two
# methods: remove_null_projects() and determine_success().
# Table with columns: project_id, prob, risk, budget, null, success
