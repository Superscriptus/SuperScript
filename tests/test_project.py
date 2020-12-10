import unittest
from unittest.mock import patch

from superscript_model.project import (Project,
                                       ProjectInventory,
                                       ProjectRequirements)


class TestProject(unittest.TestCase):

    @patch('superscript_model.organisation.TeamAllocator')
    def test_init(self, mock_allocator):

        project = Project(ProjectInventory(mock_allocator), 42, 5)
        self.assertEqual(project.project_id, 42)
        self.assertEqual(project.length, 5)
        self.assertEqual(project.progress, 0)

    @patch('superscript_model.organisation.TeamAllocator')
    def test_advance(self, mock_allocator):

        project = Project(ProjectInventory(mock_allocator), 42, 5)
        project.advance()
        self.assertEqual(project.progress, 1)
        project.advance()
        project.advance()
        self.assertEqual(project.progress, 3)

    @patch('superscript_model.organisation.Team')
    @patch('superscript_model.organisation.TeamAllocator')
    def test_terminate(self, mock_allocator, mock_team):

        inventory = ProjectInventory(mock_allocator)
        inventory.create_projects(1)
        project = inventory.projects[0]
        project.team = mock_team

        project.advance()
        project.advance()
        project.advance()
        project.advance()
        project.advance()
        self.assertEqual(mock_team.remove_lead.call_count, 1)
        self.assertIs(project.team, None)


class TestProjectInventory(unittest.TestCase):

    @patch('superscript_model.organisation.TeamAllocator')
    def test_init(self, mock_allocator):

        inventory = ProjectInventory(mock_allocator)
        self.assertEqual(inventory.active_count, 0)

    @patch('superscript_model.organisation.TeamAllocator')
    def test_create_projects(self, mock_allocator):

        inventory = ProjectInventory(mock_allocator)
        inventory.create_projects(5)
        self.assertEqual(inventory.active_count, 5)

    @patch('superscript_model.organisation.TeamAllocator')
    def test_add_project(self, mock_allocator):

        inventory = ProjectInventory(mock_allocator)
        project = Project(42)
        self.assertEqual(inventory.active_count, 0)
        inventory.add_project(project)
        self.assertEqual(inventory.active_count, 1)
        self.assertRaises(KeyError, inventory.add_project, project)

    @patch('superscript_model.organisation.TeamAllocator')
    def test_delete_project(self, mock_allocator):

        inventory = ProjectInventory(mock_allocator)
        inventory.create_projects(5)
        self.assertRaises(KeyError, inventory.delete_project, 42)
        self.assertEqual(inventory.active_count, 5)
        inventory.delete_project(0)
        inventory.delete_project(2)
        self.assertEqual(inventory.active_count, 3)

    @patch('superscript_model.organisation.TeamAllocator')
    def test_advance_projects(self, mock_allocator):

        inventory = ProjectInventory(mock_allocator)
        inventory.create_projects(10)
        for i in range(5):
            inventory.advance_projects()
        self.assertEqual(inventory.active_count, 0)


class TestProjectRequirements(unittest.TestCase):

    def test_init(self):
        r = ProjectRequirements()

        self.assertEqual(r.max_budget_increase, 0.25)
        self.assertTrue(r.risk in [5,10,25])
        self.assertTrue(r.flexible_budget in [True, False])
        self.assertTrue(r.creativity in [1,2,3,4,5])

    def test_assign_skill_requirements(self):
        r = ProjectRequirements()
        #r.assign_skill_requirements()
        self.assertEqual(sum([s['units'] for s in r.hard_skills.values()]),
                         r.total_skill_units)

    def test_to_string(self):
        r = ProjectRequirements()
        self.assertIsInstance(r.to_string(), str)


if __name__ == '__main__':
    unittest.main()
