# 2024-09-30
- batch size: 50
- testing on few batches from test (then average of them)
- training 1:
    - `learning_rate=0.001`
    - `batch_size=50`
    - `epochs=1`
    - the CC was equal to approx:   `-0.02`
- training 2:
    - `learning_rate=0.001`
    - `batch_size=50`
    - `epochs=2`
    - the CC was equal to approx:   `0.03`
- training 3:
    - `learning_rate=0.0001`
    - `batch_size=50`
    - `epochs=2`
    - the CC was equal to approx:   `-0.02`

- conclusion:
    - maybe increase learning rate
    - probably more epochs
    - might be good idea to change the metrics (better correlation coeficient)


# 2024-10-01
- three batches for test
- --num_epochs=10
    - not all -> we killed some processes earlier due the computational cost

## Variant results:

### --learning_rate=0.005
Process ID: 3435082
- quickly went up end stopped at 0.1
- probably too high learning rate
- killed in iteration 6 (stayed same from 4th epoch)

### --learning_rate=0.003
Process ID: 3435091
- learns very quickly 
- reaches 0.2 as first model (peak 0.22)
- peak in 6th epoch
- until that it loses performance
- drops to 0.17

### --learning_rate=0.001
Process ID: 3435124
- best result (0.24)
- first still around 0.8 then large step to 0.24 in 7th epoch
- peak in 7th epoch
- completely lost in the last iteration (drop to 0.12)

### --learning_rate=0.00075
Process ID: 3435162
- slow increase
- killed in epoch 6 (might get better in more epochs)
- maybe it might improve in more steps

### --learning_rate=0.0005
Process ID: 3435190
- completely lost
- tested on 4 epochs still under 0
- probably learning rate too low to learn effectively

### --learning_rate=0.0003
Process ID: 3435196
- completely lost
- tested on 4 epochs still under 0
- probably learning rate too low to learn effectively

### --learning_rate=0.0001
mv: nelze získat informace o 'output_dir/test_1//output_temp.log': Adresář nebo soubor neexistuje
Process ID: 3435228
- looks completely lost
- tested on 4 epochs still around 0
- probably learning rate too low to learn effectively


## Conclusion
- there might be some misguided values due the usage of only 3 batches for test
    - we do not expect high changes though (based on previous tests)
- it looks that the ideal learning rate is around `0.003 - 0.00075`
- the peak CC computed is `0.24`