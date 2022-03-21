import unittest
from unittest.mock import patch

from superscript_model.project import (Project,
                                       ProjectInventory,
                                       ProjectRequirements)
from superscript_model.model import SuperScriptModel


class TestProject(unittest.TestCase):

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_init(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        project = Project(
            ProjectInventory(mock_allocator, model=mock_model), 42, 5
        )
        self.assertEqual(project.project_id, 42)
        self.assertEqual(project.length, 5)
        self.assertEqual(project.progress, 0)

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_advance(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        project = Project(
            ProjectInventory(mock_allocator, model=mock_model), 42, 5)
        project.advance()
        self.assertEqual(project.progress, 1)
        project.advance()
        project.advance()
        self.assertEqual(project.progress, 3)

    @patch('superscript_model.organisation.Team')
    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.worker.Worker')
    @patch('superscript_model.model.SuperScriptModel')
    def test_terminate(self, mock_model, mock_worker,
                       mock_allocator, mock_team):

        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model)
        inventory.create_projects(1, time=0, length=5)

        project = inventory.projects[0]
        mock_team.lead = mock_worker
        project.team = mock_team

        self.assertEqual(inventory.active_count, 1)
        project.advance()
        project.advance()
        project.advance()
        project.advance()
        project.advance()
        self.assertEqual(mock_team.remove_lead.call_count, 1)
        self.assertIs(project.team, None)
        self.assertTrue(0 not in inventory.projects.keys())
        self.assertEqual(inventory.active_count, 0)
        self.assertEqual(project.success_probability, 0.0)

    def test_chemistry(self):

        model = SuperScriptModel(
            worker_count=10,
            worker_strategy='AllIn',
            organisation_strategy='Random',
            budget_functionality_flag=False
        )
        model.inventory.create_projects(1, 0, 5)
        project = model.inventory.projects[0]
        project.team.log_project_outcome(success=False)
        self.assertFalse(
            model.grid.get_team_historical_success_flag(project.team)
        )
        model.inventory.create_projects(1, 0, 5)
        project2 = model.inventory.projects[1]
        project2.team = project.team
        project2.team.log_project_outcome(success=True)
        self.assertTrue(
            model.grid.get_team_historical_success_flag(project2.team)
        )


class TestProjectInventory(unittest.TestCase):

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_init(self, mock_model, mock_allocator):

        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model,
                                     save_flag=True)
        self.assertEqual(inventory.active_count, 0)
        self.assertEqual(inventory.all_projects, {})

        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model,
                                     load_flag=True)
        self.assertEqual(inventory.load_flag, False)  # as no project file to be found

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_create_projects(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model,
                                     save_flag=True)
        inventory.create_projects(5, time=0, length=5)
        self.assertEqual(inventory.active_count, 5)
        self.assertIsInstance(inventory.all_projects[0][0], Project)

        inventory.load_flag = True
        inventory.save_flag = False
        for i in range(5):
            inventory.delete_project(i)
        inventory.create_projects(5, time=0, length=5)
        self.assertEqual(inventory.active_count, 5)

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_save_projects(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model,
                                     save_flag=True)
        inventory.create_projects(5, time=0, length=5)
        inventory.save_projects()
        self.assertEqual(inventory.active_count, 5)

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_add_project(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model)
        project = Project(inventory, 42)
        self.assertEqual(inventory.active_count, 0)
        inventory.add_project(project)
        self.assertEqual(inventory.active_count, 1)
        self.assertRaises(KeyError, inventory.add_project, project)

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_delete_project(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model)
        inventory.create_projects(5, time=0, length=5)
        self.assertRaises(KeyError, inventory.delete_project, 43)
        self.assertEqual(inventory.active_count, 5)
        inventory.delete_project(0)
    #     inventory.delete_project(2)
    #     self.assertEqual(inventory.active_count, 3)

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_advance_projects(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model)
        inventory.create_projects(10, time=0, length=5)
        for i in range(5):
            inventory.advance_projects()
        self.assertEqual(inventory.active_count, 0)

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_rank_projects(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model)
        projects = []
        for i in range(10):
            projects.append(
                Project(inventory, i, 5)
            )
        projects = inventory.rank_projects(projects)

        risk = [p.risk for p in projects]
        risk_copy = risk[:]
        risk_copy.sort(reverse=True)
        self.assertEqual(risk, risk_copy)

        risk_unique = set(risk)
        creativity = [p.creativity for p in projects]
        for r in risk_unique:

            indices = [i for i, e in enumerate(risk) if e == r]
            creativity_subset = [creativity[i] for i in indices]
            creativity_subset_copy = creativity_subset[:]
            creativity_subset_copy.sort(reverse=True)
            self.assertEqual(creativity_subset,
                             creativity_subset_copy)


class TestProjectRequirements(unittest.TestCase):

    def test_init(self):
        r = ProjectRequirements()

        self.assertTrue(r.risk in [5,10,25])
        self.assertTrue(r.flexible_budget in [True, False])
        self.assertTrue(r.creativity in [1,2,3,4,5])

    def test_assign_skill_requirements(self):
        r = ProjectRequirements()
        self.assertEqual(sum([s['units'] for s in r.hard_skills.values()]),
                         r.total_skill_units)

    def test_to_string(self):
        r = ProjectRequirements()
        self.assertIsInstance(r.to_string(), str)


class TestSuccessCalculator(unittest.TestCase):

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_determine_success(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model)
        project = Project(inventory, 42)
        project.success_probability = 1
        self.assertTrue(
            inventory.success_calculator.determine_success(project)
        )
        project.success_probability = 0
        self.assertFalse(
            inventory.success_calculator.determine_success(project)
        )

    def test_calculate_success_probability(self):

        N = 20
        model = SuperScriptModel(1000)
        model.inventory.create_projects(N, 0, 5)

        for i in range(N):
            project = model.inventory.projects[i]
            model.inventory.success_calculator.calculate_success_probability(
                project
            )

            self.assertTrue(project.success_probability >= 0)
            self.assertTrue(project.success_probability <= 1)

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_get_component_values(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model)
        project = Project(inventory, 42)
        project.team = None
        inventory.success_calculator.get_component_values(project)
        self.assertEqual(inventory.success_calculator.ovr, 0.0)
        self.assertEqual(inventory.success_calculator.risk, 0.0)

    @patch('superscript_model.organisation.TeamAllocator')
    @patch('superscript_model.model.SuperScriptModel')
    def test_to_string(self, mock_model, mock_allocator):
        mock_model.p_budget_flexibility = 0.25
        inventory = ProjectInventory(mock_allocator,
                                     model=mock_model)
        project = Project(inventory, 42)
        self.assertIsInstance(
            inventory.success_calculator.to_string(project),
            str
        )


if __name__ == '__main__':
    unittest.main()
