# Running SuperScript on AWS

### Main resources.
The SuperScript account on AWS has account number 333786391507. Data are stored in an S3 bucket called **'superscriptsotrage'**. Simulations can be run on EC2 on the following instances:
 
 * **'test'** is a t2.micro instance with a single vCPU that is included in the free tier. This can be used to run test 
 simulations. Since it only has a single thread it can only run SuperScript with ```ORGANISATION_STRATEGY = 'Random'``` or ```'Basic'``` 
 as these strategies do not require multiprocessing. Alternatively, if using the basinhopping optimisation (`````'Basin'`````), you must use 
 ```NUMBER_OF_PROCESSORS = 1```.   
 * **'parallel_simulation_machine'** is a preconfigured instance of c62.2xlarge. This is a compute optimised 
 instance with 8 vCPUs for running SuperScript simulations with basinhopping optimisation using 8 parallel processes 
 (```NUMBER_OF_PROCESSORS = 8```). Since this instance is fully configured, it is best to leave it open and just 'start' and 'stop' 
 it as required. There is no charge while the instance is stopped and the storage drive persists between sessions. **Do not terminate this instance.**   
* There is also a launch template - **'parallel_simulation_template'** with predefined a user data script from which new instances can be created. The user 
data script will automatically configure the new instance to match 'parallel_simulation_machine' by installing python etc. 
However, the SuperScript code must be manually pulled from github and a virtualenvironment created requirements file (see instructions below).     

To interact with the S3 and EC2 resources, an [IAM user account must be created](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console) and added to the group 
'simulation_runners'. This will give the IAM user the correct permissions.

As standard, the username for connecting to all AWS instances is **'ec2-user'**.

### Running simulations.

To run a simulation, start the 'parallel_simulation_machine' instance from within the AWS console (website). You can then 
connect to the instance, which will give you a terminal session within the console. You then need to:
* activate the virtual environment using ```source venv\bin\activate``` 
* set desired configuration in ```config.py```
* start a screen session (so that the simulation can be run in the background and will not stop when you disconnect) using ```screen``` 
* run the simulator script using ```python3.8 asw_run_simulation.py ./simulation_io/<subdir> 100 100 2``` (see script for parameters).
* note that you can send the output to a text file using ```python3.8 asw_run_simulation.py 100 100 2 >> output.txt```
* disconnect from the screen session by typing ```ctrl + a``` then ```d```
* to later connect to the screen session use ```screen -r```
* while the simulation is running ```top``` should show all cores in use   
 
 Instead of connecting to the instance via the online AWS console, you can connect directly from your computer via an 
 ssh session (e.g. use Putty in Windows).
 
 ### Saving data to and from S3.
 
 To check your connection to S3, while connected to the instance use ```aws s3 ls s3://superscriptstorage```.
 
 To copy a whole directory to S3 use ```aws s3 sync <directory> s3://superscriptstorage/<directory>/```.
 
 Or you can copy individual files using ```aws s3 cp```.
 
 To download data from S3, you can select individual files to download from within the AWS console. Alternatively, if you 
 have the AWS CLI installed locally you can use the same commands ```asw s3 sync``` and ```aws s3 cp``` to sync and copy 
 data to your machine.     
 
 ### Creating and configuring a new EC2 instance.
 
 Note: this assumes that you are not using a launch template. If you are using the 'parallel_simulation_template' you can
 skip steps 1 to 4 as these should be handled for you.
 
 1. Follow the procedure for creating a new instance in the AWS console, using all the default settings except for the following:
    * select instance type 'c6g.2xlarge' (or another ARM based compute optimised instance type if needed)
    * select the ARM version of the amazon free linux image on the AMI page (instead of default x86) 
    * ensure that 'ec2_full_access_to_s3' is added as an IAM role to the instance
    * uncheck the box to delete storage on instance termination.
    * create a new security group, replacing 0.0.0.0 with your public IP address (or use existing launch wizard security 
    group if you have already defined one for this IP address). 
 2. Download the new key pair when launching this instance and save it in a secure location. If using Putty, convert the 
 pem key to a putty key using puttygen.
 3. Connect to the instance, either from within the AWS console or from Putty using the key you just downloaded 
 ([instructions here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html)).
 4. Configure the instance:
    * ```sudo yum update -y```
    * ```sudo yum install git -y```
    * ```sudo amazon-linux-extras install python3.8 -y```
    * ```sudo yum groupinstall "Development Tools" -y```
    * ```sudo yum install python38-devel -y```
    * ```sudo yum install blas-devel lapack-devel -y```
    * ```curl -O https://bootstrap.pypa.io/get-pip.py```
    * ```python3.8 get-pip.py```
    * ```pip install --upgrade pip```
    * ```python3.8 -m venv /home/ec2-user/venv```
    * ```source /home/ec2-user/venv/bin/activate```
 5. Install SuperScript:
    * ```git clone https://github.com/cm1788/SuperScript.git``` (will need to authenticate)
    * ```cd SuperScript```
    * ```pip install -r requirements_pip.txt```
           
  