# TechAdoption

version 1.0 released on May 30, 2020

## Description

TechAdoption allows for the identification of top features predictive of technology adoption and creates a 
model to predict adoption. The software package interfaces with Magpi generated datasets 
created using the Ethnographic Decision Model (EDM) methodology. The model can also be tested uses other datasets
not used to build the model. For more information on the EDM methodology, please refer to "Ethnographic Decision Tree Modeling"
Book by Christina H. Gladwin and the fifth edition of "Research Methods in Anthropology: Qualitative and Quantitative Methods" by 
H. Russel Bernard. 


## Requirements

TechAdoption is a python software package. I recommend downloading [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 
that includes python and all necessary dependencies to run TechAdoption. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install TechAdoption from the Anaconda command prompt.

```bash
pip install TechAdoption
```

## Usage
To build the model, run Build_Model.py from your current working directory (cwd) with the number of devices or options in your dataset 
(nd) and number of questions asked per device or option (nq) which are mandatory inputs. Example below for 4 devices/options and 30 questions. 

```python

cwd\python Build_Model.py -nd 4 -nq 30

```

For other optional inputs, use the help command.

```python

cwd\python Build_Model.py -help

```

After running the above line, a window will pop up allowing you to select the Magpi csv file to build the model.

Outputs from running this code include:
1. the accuracy of the model
2. a printed list of features influencing adoption in descending order of importance
3. bar chart with the features in descending order of important (Plot_Variable_Importance.png)
4. scatter plot with predicted versus actual data and corresponding accuracy (Plot_Predicted_Actual.png)
5. one random forest tree for illustrative purposes (Plot_tree.png) - model and variable importances are based on the
average of 1000 (default) of these trees


To test the model using an independent dataset, run Predict_Model.py from your current working directory with the number of 
devices in your dataset (nd) and number of questions asked per device (nq) which are mandatory inputs. Example below for
4 devices and 30 questions. Again, the same optional inputs are available for input at this time as well.

```python

cwd\python Test_Model.py -nd 4 -nq 30

```

After running the above line, a window will pop up allowing you to select the Magpi csv file to build the model. A 
second window will then pop up allowing you to then select the Magpi csv file to test the model.

## Example
To try running an example, run the following line of code.

``` python

cwd\python Build_Model.py -nd 8 -nq 12

```
When the window pops up to select a csv file, choose "Magpi_Dummy_data.csv" from the "example" folder inside the software package.
The outputs of the model and plots should be similar to those in the "example" folder. It is important to note that random forests 
are "random" so results may vary a bit, but overall trends should be consistent. 

## Testing
The Test folder contains the Test_Build_Model.py. This file runs various tests on the functions in the Build_Model.py file. If any changes
are made to the code, please varify that all tests pass. To run tests, set your working directory to the outermost level of the package
and run the below line of code.

```python

cwd\pytest

```

## Support
Please email peiffer.erin@gmail.com if you have any questions or would like further support. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authorship
Created by Erin Peiffer, Spring 2020 

To cite: 
Peiffer, E (2020) TechAdoption (Version 1.0) [Software package]. 

- Web address or publisher (e.g. program publisher, URL)

## License
[MIT](https://choosealicense.com/licenses/mit/)

