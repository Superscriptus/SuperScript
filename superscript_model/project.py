class ProjectInventory:

    def __init__(self):
        self.projects = dict()
        self.index_total = 0

    @property
    def active_count(self):
        return len(self.projects.keys())

    def create_projects(self, new_projects_count):

        for i in range(new_projects_count):
            p = Project(self, self.index_total + i)
            self.projects[p.project_id] = p
            self.index_total += 1

    def advance_projects(self):
        """ Allows projects to be deleted/terminated during loop"""
        project_ids = list(self.projects.keys())

        for pid in project_ids:
            if pid in self.projects.keys():
                self.projects[pid].advance()


class Project:

    def __init__(self,
                 inventory: ProjectInventory,
                 project_id=42,
                 project_length=5):

        self.inventory = inventory
        self.project_id = project_id
        self.length = project_length
        self.progress = 0

    def advance(self):
        self.progress += 1
        if self.progress == self.length:
            self.terminate()

    def terminate(self):
        del self.inventory.projects[self.project_id]


