from superscript_model.model import SuperScriptModel

import matplotlib.pyplot as plt

abm = SuperScriptModel()
abm.run_model(100)

fs = 12
ps = 5

projects = abm.datacollector.get_table_dataframe("Projects")

print(sum(projects.prob==0.0))

null_projects = projects.loc[projects.null]
projects = projects.loc[~projects.null]
successful_projects = projects.loc[projects.success]
failed_projects = projects.loc[~projects.success]
#print(null_projects)
print(len(null_projects))


plt.scatter(successful_projects.budget,
            successful_projects.prob * successful_projects.risk,
            label='successful projects', color='green', s=ps)

plt.scatter(failed_projects.budget,
            failed_projects.prob * failed_projects.risk,
            label='failed projects', color='red', s=ps)

plt.scatter(null_projects.budget,
            null_projects.prob * null_projects.risk,
            label='null project', color='orange', s=ps)

plt.xlabel("project budget", fontsize=fs)
plt.ylabel("expected value (prob * risk)", fontsize=fs)
plt.title("SuperScript model with %d workers" % abm.worker_count)
plt.legend()
plt.savefig("scatter_example.png")
plt.show()
