# Simple model

## 2024-10-03
Batch size: 50
Num epochs: 10

### output_3688069.log
Learning rate: 0.003
- max correlation reached 0.11
- stayed roughly the same from 7th epoch
- probably too large learning rate

### output_3688079.log
Learning rate: 0.001
- max correlation reached 0.18
- much faster learning (maximum in 4th epoch)
- lately dropped to 0.14 (still descending)
    - probably overfitted

### output_3688152.log
Learning rate: 0.00075
- max correlation reached 0.22
- relatively slow start (till 5th epoch 0.06)
    - then rapid ascend in 2 epochs (7th is max)
- then slowly dropped to 0.18


# Complex Model
- using complex model (NN instead of simple neuron) 

## 2024-10-05
Batch size: 50
Num epochs: 6
    - we decided to kill the processes as we did not saw significant improvement in the learning
Complexity model size: 64

### output_3840615.log  
Learning rate: 0.003
- CC still fluctuated around 0
- we do not see any significant improvement in the learning

### output_3840663.log 
Learning rate: 0.001
- CC still fluctuated around 0.02
- we do not see any significant improvement in the learning
    - a bit better than 0.003 LR (still no improvement seen)

### output_3840667.log
Learning rate: 0.00075
- CC still fluctuated around 0.02
- we do not see any significant improvement in the learning
    - probably the best option (still weak)
