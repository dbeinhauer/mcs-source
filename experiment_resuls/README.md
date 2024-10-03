# About
This directory contains results of training the model in different setups.
Each subdirectory specifies the size of the model in percentage of the original one.

# Experiment results

## Size 10
- it might be good idea to change the metrics to work with discrete values (Pearson's CC does not)
- from the first experiments it looks that ideal 
    - `learning rate` is probably around `0.003 - 0.00075`
    - `number of epochs` probably around `6 - 10`
        - or higher and we did not test it 
            - not very probable
            - if more then very time consuming (almost impossible to learn for us)

