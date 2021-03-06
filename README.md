<!--
*** README template from project: https://github.com/othneildrew/Best-README-Template
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU APGLv3 License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/cm1788/SuperScript">
    <img src="documentation/images/logo.svg" alt="Logo" width="390" height="80">
  </a>

  <h3 align="center">Simulation engine for a social teamwork game</h3>

  <p align="center">
    SuperScript is an agent-based model of team formation in large organizations, built using the 
    <a href="https://github.com/projectmesa/mesa">Mesa</a> framework.
    <br />
    <a href="https://github.com/cm1788/SuperScript"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/cm1788/SuperScript/issues">Report Bug</a>
    ·
    <a href="https://github.com/cm1788/SuperScript/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#analysis">Analysis</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#tests">Tests</a></li>
    <li><a href="#model-development">Model development</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![Product Name Screen Shot][product-screenshot]

The purpose of the extended model is to build a simulation engine that serves as the basis for the “social teamwork game”. The model simulates the state of the world and its development over time for workers within an organization that are assembled in teams to work on projects. 
The simulation engine will then serve as the underlying model for the world in a Python-Django prototype for a web application (e.g., hosted on Heroku). In that prototype a user (either a worker or an organization) goes through the steps of the user journey and takes part in the social teamwork game. 
The model will be used to test a number of hypotheses to understand emergent properties and/or to fine-tune the input parameters, to visualize those results, to generate training data for the Machine Learning prototype and to gather model proof points for further discussion. 


### Built With

SuperScript is built with the following key frameworks/packages:
* [Mesa](https://github.com/projectmesa/mesa): agent-based modeling framework.
* [Scipy.optimize](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html): optimisation package used for 
team allocation.
* [Pathos](https://pypi.org/project/pathos/): used for parallel optimizations on multicore architectures.



<!-- GETTING STARTED -->
## Getting Started

It should be straightforward to get SuperScript working on any system that has Python installed.
To get a local copy up and running follow, these simple steps.

### Prerequisites

The following are required in order to run SuperScript locally:
* [Python3.6 or above (python_version>="3.6")](https://www.python.org/downloads/)
* either [pip](https://pip.pypa.io/en/stable/installing/) (recommended) or 
[conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
* [venv](https://docs.python.org/3/library/venv.html)  



### Installation

The recommended installation method is to use the package manager [pip](https://pip.pypa.io/en/stable/installing/),
because this is the most lightweight solution on new machines (e.g. on AWS). But instructions are also provided for 
installation using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).



#### Using pip:   

1. Clone the repo and change into the directory:
   ```
   git clone https://github.com/cm1788/SuperScript
   
   # Windows:
   chdir SuperScript
   
   # Linux:
   cd SuperScript
   ```
   
   _Note: if you have an authentication error when using ```git clone``` (or you get a ```repository not found``` error), 
   you can use the syntax: ```git clone https://username:password@github.com/cm1788/SuperScript```_
    
2. Create a new virtual environment: 
    ```
    python -m venv superscriptenv
   ```
   _Note: the ```python``` command here may need to be replaced with ```python3``` or 
    ```python3.6``` or ```py``` depending on your system, whichever points to the 
    version of python that you want to use._
3. Activate your virtual environment:  
    ```
   # Windows:
   superscriptenv\Scripts\activate
   
   # Linux:
   source superscriptenv/bin/activate
   ```
4. Install requirements (ensuring that you use the correct requirements file for windows/linux):
    ```
    python -m pip install --upgrade pip
   
    # Windows:
    python -m pip install -r requirements.txt
   
   # Linux:
   python -m pip install -r requirements_linux.txt
   ```
   _Note: just using ```python``` here should be fine provided that you have activated the ```superscriptenv``` 
   environment._
   
   _Note: The requirements.txt file was produced using ```pip list --format=freeze > requirements.txt``` on a linux 
   system with Python3.6. Depending on your installation system the following dependencies may have problematic version
   specifications: ```kiwisolver, numpy, scipy```. If these or any other dependencies fail to install, try manually 
   removing the version specification from ```requirements.txt``` (i.e. remove the ```==X.X.X```from that line in the 
   file)._    


#### Using conda:   

1. Clone the repo
   ```
   git clone https://github.com/cm1788/SuperScript
   ```
2. Create virtual environment from the YAML file: 
    ```
    conda env create -f superscriptenv.yml
   ```
3. Activate the virtual environment:
   ```
   conda activate superscriptenv
   ```


<!-- USAGE EXAMPLES -->
## Usage

To run SuperScript in the Mesa server, first ensure that the ```superscriptenv``` environment is activated and then use:
```
mesa runserver 
```

Most of the important model parameters can be selected in the GUI, but for other parameters they will need to be 
adjusted in [config.py](superscript_model/config.py) before launching 
the server.

To activate the social network visualisation, you need to uncomment line 318 in 
[server.py](superscript_model/server.py) (this feature is deactivated 
by default because it is slow to recompute the network layout on each timestep). However, the social network can be 
saved for later analysis by setting the ```SAVE_NETWORK``` flag to ```True```.

_Note: The parallel basinhopping optimisation (```ORGANISATION_STRATEGY = Basin```) can be very slow depending on the 
size of simulation. For real-time visualisation it is best to use ```Random``` or ```Basic```, although the teams 
produced will not be as good._

#### Running simulations on AWS

Instructions for getting set up on AWS are provided in 
[documentation/aws_instructions](documentation/aws_instructions.md) 
and there is a python script provided for running these simulations: 
[aws_run_simulations.py](aws_run_simulation.py)

#### Running batch simulations

A python script for running batch simulations is provided: [batch_run_simulation.py](batch_run_simulation.py)

### Analysis

There are five analysis notebooks and the data to run them are include in the repository 
(in [analysis/](analysis)):
1. [initial_aws_simulations.ipynb](analysis/initial_aws_simulations.ipynb) explores early simulated data (pre-release).
2. [testing_hypotheses.ipynb](analysis/testing_hypotheses.ipynb) explores batch simulation data the were produced using 
release v0.1 of the model.
3. [testing_hypotheses_2.ipynb](analysis/testing_hypotheses_2.ipynb) expands on the previous analysis to study the effects
   of the skill decay parameter.
4. [network_analysis.ipynb](analysis/network_analysis.ipynb) studies the social network of successful collaborations 
   between workers.
5. [probability_components.ipynb](analysis/probability_components.ipynb) explores the project success probabilities and 
   their components and develops a probability normalisation method that is used in releases v1.2 and above to ensure 
   that success probabilities lie in the closed interval [0, 1].   

The hypotheses are detailed in the [model specification document](documentation/model_specification.pdf). There are 
also new analysis scripts in [retrospective_metrics](analysis/retrospective_metrics). These scripts are for metrics or 
analyses that are currently (v1.2) being computed retrospectively, however in the next release they will be incorporated
 into the core simulation model, and the scripts will be deprecated (see Roadmap below). The scripts are:
1. [network_reconstruction_ne.py](analysis/retrospective_metrics/network_reconstruction_ne.py) which reconstructs the 
network of successful collaborations between workers on every timestep and stores this in a concise representation (an
   initial and diff file) to be imported to networkx. 
2. [preset_E.py](analysis/retrospective_metrics/preset_E.py) which creates a new pseudo-simulation by post-processing
   pre-simulated data to removing the inactive workers from the workforce on each timestep (down to a target slack of
   10%). The resulting simulation is referred to as 'Preset E' in the Streamlit application.
3. [roi.py](analysis/retrospective_metrics/roi.py) which computes the new 'Return on Investment' metric and averages it
   across the workforce on each timestep. See the [model specification document](documentation/model_specification.pdf)
   for more details.

### Roadmap

The current roadmap consists of the following milestones  with expected
shipping dates in square brackets:

1. Add ROI calculation, 'Preset E' and new network output format to the core simulation code (rather than using scripts 
   to retrospectively compute these.) [Q1 2022]  
2. Complete unit testing framework to reach 100% (currently at 97%). [Q1 2022]
3.	Add performance benchmarking (for optimisation routine). [Q3 2021]
4. Implement Reinforcement Learning approach to team allocation and compare performance with basin-hopping. [Q2 2022]
5.	Alter training mechanism (and explore the 'rich get richer' effect). [tbc]
6.	Edit social network to track more information about historical collaborations. [tbc]
7.	Review model extensions and choose those to implement. [tbc]

See [open issues](https://github.com/Superscriptus/SuperScript/issues) for details of milestone 1. 

### Tests

The repository is currently at 97% code coverage. To run the unit tests use:
```
coverage run -m unittest discover && coverage report
```
For an interactive html coverage report run ```coverage html -i``` and then open ```index.html``` in your browser.


### Model development

The [documentation](./documentation) folder contains a word document with the full 
[model specification](documentation/model_specification.pdf).

The directory [model_development](./model_development) contains Jupyter Notebooks relating to various stages of 
development of the model, from initial experiments prior to coding to the model, through to integration tests and 
performance benchmarking. These notebooks include:

* [active_project_equilibrium.ipynb](model_development/active_project_equilibrium.ipynb): short experiment to determine 
equilibrium behaviour of the model based on proposed dynamics (i.e. number of active projects, number of active workers, 
number of workers per project). Written prior to coding up the model.
* [function_definitions.ipynb](model_development/function_definitions.ipynb): contains definitions of the various 
functions used throughout the model, with the original piecewise functions that were proposed in the draft spec, and 
possible alternatives that were suggested. In the model code functions are supplied by the 
[FunctionFactory](superscript_model/function.py).    
* [go_allocate_experiment.ipynb](model_development/go_allocate_experiment.ipynb): determines the time taken to find the 
global optimum team allocation (which was originally called 'go_allocate'). The conclusion, as expected, is that the 
problem is not computationally tractable for large simulations, and so numerical optimisation will be required.
* [manually_testing_model.ipynb](model_development/manually_testing_model.ipynb): simple integration test to check that 
model is running correctly and inspect some of the variables.
* [comparing_strategies.ipynb](model_development/comparing_strategies.ipynb): comparing the 'Random' and 'Basic' team 
allocation strategies to confirm that 'Basic' does improve over the random method in terms of project success 
probability.    
* [optimization_experiments_gekko.ipynb](model_development/optimization_experiments_gekko.ipynb): failed attempt to get 
Gekko mixed-integer optimisation working (included for completeness).
* [optimization_experiments_scipy.ipynb](model_development/optimization_experiments_scipy.ipynb): experiments to get 
SciPy optimisation up and running. 
* Benchmarking notebook: to be added in future milestone. 



<!--
## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->


<!-- ACKNOWLEDGEMENTS 
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
-->


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Superscriptus/SuperScript.svg?style=for-the-badge
[contributors-url]: https://github.com/Superscriptus/SuperScript/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Superscriptus/SuperScript.svg?style=for-the-badge
[forks-url]: https://github.com/Superscriptus/SuperScript/network/members
[stars-shield]: https://img.shields.io/github/stars/Superscriptus/SuperScript.svg?style=for-the-badge
[stars-url]: https://github.com/Superscriptus/SuperScript/stargazers
[issues-shield]: https://img.shields.io/github/issues/Superscriptus/SuperScript.svg?style=for-the-badge
[issues-url]: https://github.com/Superscriptus/SuperScript/issues
[license-shield]: https://img.shields.io/github/license/Superscriptus/SuperScript.svg?style=for-the-badge
[license-url]: https://github.com/Superscriptus/SuperScript/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/michaelchristen/
[product-screenshot]: documentation/images/screenshot.png