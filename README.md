# The Impact of Macroeconomic Parameters and Global Markets on Forecasting the FTSE Index
*author: Patrick Buhagiar*

*date: 11/09/2018*

## Installation requirements
* Install python 3.6+
* Using `pip3`, install:
    * `numpy`
    * `pandas`
    * `tensorflow`

## How to run
All the experiments are in the `src` folder. Each expirment is named after the requirements in Chapter 3. To run the code, execute in command promt or console.

```
python3 h1.py
```
Replace with relevant experiment. Each experiment outputs two `.csv` files, one for accuracy and another for parameters. These files can the be read by the `result_plotter.py` file, after changing the appropriate parameters and file name in the file. 

Stage 1 of h2 and h3 persist models under the `h2_models` and `h3_models`. These folders already consist of the models created in this experiment, thus stage 1 can be skipped. 

Finally, `the unparametric_test.py` is used to output the predictions for each model. these predictions, found under the `prediction` folder, can be used for the Kruskall-Wallis test and Wilcoxon test. 

## Other notes
These experiments were run on Google Cloud, with VMs set up with 8-20 cores, thus they were meant to run on several threads and cores. 

