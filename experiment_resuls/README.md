# About
This directory contains results of training the model in different setups.
Each subdirectory specifies the size of the model in percentage of the original one.


# Experiment results

## Size 10
- best CC:  `0.84`
- the results are quite random in case of training
    - there is very high fluctuation during the training (rapid grows/drops)
- generally it is good to use small learning rates (under `5e-05`)
- regarding number of epochs it is variable
    - sometimes it is high at the begging sometimes it reached maximum after last epoch
- complex models look much more stable
    - there is some pattern in CC development

## Size 25
- generally slower training and worse results in comparison to smaller model

### Simple Model
- looks very unstable and sometimes learns quickly, sometimes pretty badly


### Complex model
- learning looks much more stable in comparison to simple model


## Size 50
