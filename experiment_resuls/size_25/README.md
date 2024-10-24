# Simple

## lr_0.001-0.00075 
- run for 10 epochs
- learning rates: `[0.001, 0.00087, 0.00075]`
- it does not learn at all
- it usually started around CC `0.085`
- ends around `0` -> definitely not learning (there is gradual descend)

## lr_0.002-0.0001
- run only 4 epochs
- learning rates: `[0.002, 0.0005, 0.0001]`
- it does not seem to be learning except the lowest learning rate which reached `0.24`


## lr_0.0002-5e-05 
- run for 9 or 10 epochs 
- learning rates:   `[0.0002, 0.0001, 5e-05]`
- the results does not look very promising but much better than larger learning rates
- only reasonably good result was for lr `5e-05`
    - max CC: `0.34`
    - start was very fluctuating (around `-0.2` to `0.2`)
    - after epoch 5 it gradually grew to peak in epoch 9 -> then drop to `0.26`

# Complex

## output_1485954.log
- learning rate: `5e-05`
- num epochs: 8
- residual connection: `True`
- max CC: `0.65` (epoch 3)
    - starts pretty high at `0.46`
    - reached maximum at epoch 3 and then drops down to `0.41`


