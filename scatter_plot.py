from superscript_model.model import SuperScriptModel

import matplotlib.pyplot as plt
import pandas as pd

abm = SuperScriptModel()
abm.run_model(20)

# We are going to use the DataCollector to collect information on project from two
# methods: remove_null_projects() and determine_success().
# Table with columns: project_id, prob, risk, budget, null, success
projects = abm.datacollector.get_table_dataframe("Projects")

#projects.dropna(subset=['budget'], inplace=True)
projects = projects.loc[projects.null==False]
print(projects)

#plt.scatter(projects.prob * projects.risk, projects.budget)
#plt.show()
