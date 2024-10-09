# About
This directory contains results of training the model in different setups.
Each subdirectory specifies the size of the model in percentage of the original one.


# Experiment results

## Size 10
Best CC is `0.24`
- it might be good idea to change the metrics to work with discrete values (Pearson's CC does not)
- from the first experiments it looks that ideal 
    - `learning rate` is probably around `0.003 - 0.00075`
    - `number of epochs` probably around `6 - 10`
        - or higher and we did not test it 
            - not very probable
            - if more then very time consuming (almost impossible to learn for us)


## Size 25

### Simple Model
Best CC is: `0.22`
- slightly worse results than in size 10 (only 3 examples though)
- it looks the smallest learning rate gets the best results
    - the best learnin rate `0.00075`
    - number of epochs is ideally approx `6-8`
        - after that it looks it overfits

### Complex model
Best CC is: `0.02`
- by far we do not see any significant sign of learning


## Size 50
