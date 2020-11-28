class Project:

    def __init__(self,
                 project_id: int,
                 project_length: int):

        self.project_id = project_id
        self.length = project_length
        self.progress = 0

    def advance(self):
        self.progress += 1
