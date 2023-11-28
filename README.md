# Optimal-Batched-Linear-Bandits

## How to run

### Random Example

`python main.py --seed 1  --K 3 --d 2 --T 100000  --num_sim 10 --verbose`

`python main.py --seed 1 --K 5 --d 3 --T 10000  --num_sim 10 --verbose`

`python main.py --seed 1 --K 9 --d 5 --T 10000  --num_sim 10 --verbose`

`python main.py --seed 1 --K 50 --d 20 --T 10000  --num_sim 10 --verbose`

### end of optimism

$\epsilon=0.01,0.2$

Instances: $\theta=(1,0)$, $\mathcal X=(1,0),(1-\epsilon,2\epsilon),(0,1)$

`python main.py   --seed 1 --d 2 --T 10000  --num_sim 10  --epsilon 0.01`

`python main.py   --seed 1 --d 2 --T 10000  --num_sim 10  --epsilon 0.2`

Instances: $\theta=(1,0,0)$, $\mathcal X=(1,0,0),(0,1,0),(0,0,1),(1-\epsilon,2\epsilon,0),(1-\epsilon,0,2\epsilon)$

`python main.py   --seed 1 --d 3 --T 30000  --num_sim 10  --epsilon 0.01`

`python main.py   --seed 1 --d 3 --T 30000  --num_sim 10  --epsilon 0.2`

Instances: $\theta=(1,0,0,0,0)$, $\mathcal X=(1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1),(1-\epsilon,2\epsilon,0,0,0),(1-\epsilon,0,2\epsilon,0,0),(1-\epsilon,0,0,2\epsilon,0),(1-\epsilon,0,0,0,2\epsilon)$

`python main.py   --seed 1 --d 5 --T 50000  --num_sim 10  --epsilon 0.01`

`python main.py   --seed 1 --d 5 --T 50000  --num_sim 10  --epsilon 0.2`
