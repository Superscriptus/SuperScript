import unittest

from superscript_model.project import Project


class TestProject(unittest.TestCase):

    def test_init(self):

        project = Project(42)
        self.assertEqual(project.project_id, 42)


if __name__ == '__main__':
    unittest.main()
