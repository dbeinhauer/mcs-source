# 2024-10-03
Batch size: 50
Num epochs: 10

## output_3688069.log
Learning rate: 0.003
- max correlation reached 0.11
- stayed roughly the same from 7th epoch
- probably too large learning rate

## output_3688079.log
Learning rate: 0.001
- max correlation reached 0.18
- much faster learning (maximum in 4th epoch)
- lately dropped to 0.14 (still descending)
    - probably overfitted

## output_3688152.log
Learning rate: 0.00075
- max correlation reached 0.22
- relatively slow start (till 5th epoch 0.06)
    - then rapid ascend in 2 epochs (7th is max)
- then slowly dropped to 0.18