# Description
Scripts for running time merger using metacentrum jobs.
# Usage
First, create an environment that will be shared across the jobs.
```bash
./prepare_env.sh
```
Then, run the `queue.sh` script which will submit 12 jobs.
The desired time interval needs to be supplied, for example 20 [ms].
```bash
./queue.sh 20
```
