import subprocess

subprocess.run(['python',
                '-m',
                'pip',
                'list',
                '--format=freeze',
                '>',
                'requirements.txt'])

