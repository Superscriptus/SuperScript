from .function import FunctionFactory


class ProjectInventory:

    def __init__(self, team_allocator):
        self.projects = dict()
        self.index_total = 0
        self.team_allocator = team_allocator
        self.timeline_flexibility_func = (FunctionFactory
                                          .get('TimelineFlexibility'))

    @property
    def active_count(self):
        return sum([1 for p in self.projects.values() if p.progress >= 0])

    def create_projects(self, new_projects_count):

        for i in range(new_projects_count):
            p = Project(self, self.index_total)
            self.team_allocator.allocate_team(p)
            self.add_project(p)

    def add_project(self, project):

        if project.project_id not in self.projects.keys():
            self.projects[project.project_id] = project
            self.index_total += 1
        else:
            raise KeyError('Project ID %d already exists in inventory.'
                           % project.project_id)

    def delete_project(self, project_id):
        try:
            del self.projects[project_id]
        except KeyError:
            print('Project ID %d not in inventory.' % project_id)
            raise

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
                 project_length=5,
                 start_time_offset=0):

        self.inventory = inventory
        self.project_id = project_id
        self.length = project_length
        self.progress = 0 - start_time_offset
        self.team = None

    def advance(self):
        self.progress += 1
        if self.progress == self.length:
            self.terminate()

    def terminate(self):
        if self.team is not None:
            self.team.remove_lead(self)
            self.team = None

        self.inventory.delete_project(self.project_id)


