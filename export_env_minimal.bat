@ECHO OFF
python -m pip list --format=freeze > requirements.txt
type requirements.txt | findstr /v mkl | findstr /v sip | findstr /v pywin32 > requirements.txt
conda env export > SuperScript.yml