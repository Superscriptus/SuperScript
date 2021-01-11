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
null_projects = projects.loc[projects.null]
projects = projects.loc[~projects.null]
print(null_projects)

# plt.scatter(projects.budget, projects.prob * projects.risk, label='projects that ran')
# plt.scatter(null_projects.budget,
#             null_projects.prob * null_projects.risk,
#             label='null project', color='orange')

#plt.scatter(projects.prob, projects.risk, label='projects that ran')
plt.scatter(null_projects.prob, null_projects.risk,
            label='null project', color='orange')

plt.xlabel("project budget")
plt.ylabel("expected value (prob * risk)")
plt.legend()
plt.show()
