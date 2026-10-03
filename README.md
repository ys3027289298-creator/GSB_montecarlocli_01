# Monte Carlo Simulation CLI Tool

- [Monte Carlo Simulation CLI Tool](#monte-carlo-simulation-cli-tool)
  - [🚀 Features](#-features)
  - [🛠️ Getting Started](#️-getting-started)
  - [Installation](#installation)
  - [⚠️ Warnings](#️-warnings)
  - [Version history](#version-history)

Welcome to the Monte Carlo Simulation CLI Tool repository! This project is a collection of code I’ve written over the years to simplify project forecasting using monte carlo simulations. Initially inspired by a presentation from Dimitar Bakardzhiev and fueled by the work of Daniel Vacanti and Prateek Singh, I’ve moved away from Excel-based simulations to Python for greater flexibility and scalability.
## 🚀 Features

    Flexible Simulations: Run Monte Carlo simulations to forecast project timelines.
    Command-line Interface (CLI): Easily interact with the tool via the command line.
    Customizable Parameters: Modify key parameters to fit the specific needs of your project.
    Real-world Tested: This tool has been applied in real-life project management scenarios.


## 🛠️ Getting Started
Prerequisites

    Python 3.7 or higher
    pip

## Installation

Clone the repository:

`git clone https://github.com/nlovadina/montecarlocli.git`

Prepare a python virtual environment:

`$ python3 -m venv montecarlocli`

Navigate to the project directory:

`$ cd montecarlocli`

Activate the environment

`$ . ./bin/activate`

Finally install it with pip. The tool only uses the Python standard library, so no third-party dependencies are downloaded:

`$ pip install --editable .`

And last step: 
`$ montecarlocli --help`

Have fun.

## ⚠️ Warnings 

The code needs desperatly some adjustement, cleanup and improvement because I literally took various script and put them togheter in a file.
Before using the simulation I reccomand watching the following video on youtube:

https://www.youtube.com/watch?v=A_0ngk6OlRg

https://www.youtube.com/watch?v=ykvwBmVMXBw

https://www.youtube.com/watch?v=Cdqp8029qn4

https://www.youtube.com/watch?v=beQ6qGsVLJE

https://www.youtube.com/watch?v=RTPcQrLE_kk

https://www.youtube.com/watch?v=6At_Kwjvh1Q

Because there is no silver bullet.

## Version history

1. V 0.1 **07/10/2024 First release of the code.**
2. V 0.2 **12/10/2024 Better command line.** The two simulation has been logically separated and some more options have been added read the internal help.
3. v 0.3 **13/10/2024 Multiple percentiles.** Added option to have multiple percentiles and removed dependency on colorama.
4. v 0.4 **16/10/2024 Start integration with rich package.** Added progress bar from rich package to opening of the data file and to the calculation of the simulation.
5. v 0.5 **03/10/2026 Standard library only, decoupled pipeline.** Removed the Click/numpy/pandas/rich/scipy dependencies. The random number stream, batch processing and aggregation are now independent components: use `-s/--seed` for reproducible results, and the batch size only affects progress reporting, never the statistics. Zero or negative numbers of trials are rejected, the standard deviation is now the unbiased (Bessel-corrected) sample standard deviation, and the progress bar cleans up its line when a run is interrupted.