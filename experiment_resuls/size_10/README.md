# Simple

## first_run_high_lr_range
- learning rates: `[0.001, 0.00885, 0.00075, 0.000885, 0.0005]`
- running only on 4 epochs
- generally not learning much - highest CC was around 0.43
    - majority stays around 0.22 or 0.1 (for very high lr)
        - often also worse results during training

## lr_0.002-0.0001_epochs_4
- learning rates: `[0.001, 0.00075, 0.0005, 0.0001, 0.002]`
- running only on 4 epochs
- generally not learning much
    - CC around 0.22 (and sometimes lowers during training)
    - only positive result is for lr `0.0001` -> CC: `0.74`
        - and gradually learns during training
        - probably it would be good use lr `0.0001` and lower

## output_1492886.log
- learning rate: `5e-05`
- running on 10 epoch
- currently best CC for `simple` model of size `0.10`
- best CC:  `0.82`
- gradually learns till epoch 6
    - after epoch 6 is stayed roughly the same (the worse CC after was `0.809`)


# Complex
- using smaller learning rates: `[5e-05, 3e-05, 1e-05]`
- both variant with and without residual connection:

## output_1492632.log
- lr `5e-05`
- residual: `True`
- max CC: `0.8`
    - after epoch 3
    - then gradually drops to `0.41` in the last step
    - steep jump after first epoch (correlation around `0.64`)

## output_1492685.log
- lr `5e-05`
- residual: `False`
- max CC: `0.83`
    - after epoch 10
    - first `0.1` then drop to around `0` then steep learning to `0.64` in epoch 4
    - from epoch 4 it looks it gradually learns (might make more sense than immediate jump in variant with residual connections)


## output_1492801.log 
- lr `3e-05`
- residual: `True`
- max CC: `0.74`
    - maximum in epoch 3
    - steep jump in epoch 1 (to `0.46`)
    - from epoch 3 it gradually drops to CC `0.32`

## output_1492842.log
- lr `3e-05`
- residual: `False`
- max CC: `0.84`
    - start with CC `0.15`
    - steep learning (in epoch 4 it has `0.81`)
    - slight drop to `0.68` in one epoch
    - after that it grows to values around `0.83` and stayed there till the end
    - it looks like it reached the peak of the learning in epoch 7

## output_1492753.log
- lr `1e-05`
- residual: `True`
- max CC: `0.76`
    - relatively rapid start 
        - first epoch `-0.26`
        - after second epoch `0.41`
        - max in epoch 7
        - then gradual descent

## output_1492719.log 
- lr `1e-05`
- residual: `False`
- max CC: `0.67`
    - very slow start around `0`
    - then steep drop to `-0.25` where it stays for few epochs
    - from epoch 6 it gradually grows 
        - looks like from that step it is learning