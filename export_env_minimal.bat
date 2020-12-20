@ECHO OFF
python -m pip list --format=freeze > requirements.txt
conda env export > SuperScriptMinimal.yml