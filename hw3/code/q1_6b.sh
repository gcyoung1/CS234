#!/bin/bash

for env in cartpole pendulum cheetah;
do echo $env;
   for seed in 1 2 3;
   do echo $seed;
      nohup python main.py --env-name $env --seed $seed --no-baseline &
      nohup python main.py --env-name $env --seed $seed --baseline &
   done;
done;
