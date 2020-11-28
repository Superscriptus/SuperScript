import unittest

from superscript_model.project import Project


class TestProject(unittest.TestCase):

    def test_init(self):

        project = Project(42, 5)
        self.assertEqual(project.project_id, 42)
        self.assertEqual(project.length, 5)
        self.assertEqual(project.progress, 0)

    def test_advance(self):

        project = Project(42, 5)
        project.advance()
        self.assertEqual(project.progress, 1)
        project.advance()
        project.advance()
        self.assertEqual(project.progress, 3)



if __name__ == '__main__':
    unittest.main()
