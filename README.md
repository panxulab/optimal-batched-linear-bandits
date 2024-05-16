# Optimal Batched Linear Bandits

### <p align="center">[ICML 2024]</p>

<p align="center">
  <a href="">Xuanfei Ren</a><sup>*</sup> ·
  <a href="https://tianyuanjin.github.io/">Tianyuan Jin</a><sup>†</sup> ·
  <a href="https://panxulab.github.io/">Pan Xu</a><sup>‡</sup>
</p>
<p align="center">
<sup>*</sup> University of Science and Technology of China ·  
<sup>†</sup> National University of Singapore ·
<sup>‡</sup> Duke University ·
</p>

Implementation of the paper "Optimal Batched Linear Bandits".

## How to run

### The "End of Optimism" instances

$\epsilon=0.01,0.2$

Instances: $\theta=(1,0)$, $\mathcal X=(1,0),(1-\epsilon,2\epsilon),(0,1)$

`python main.py    --d 2 --T 10000  --num_sim 10  --epsilon 0.01`

`python main.py    --d 2 --T 10000  --num_sim 10  --epsilon 0.2`

Instances: $\theta=(1,0,0)$, $\mathcal X=(1,0,0),(0,1,0),(0,0,1),(1-\epsilon,2\epsilon,0),(1-\epsilon,0,2\epsilon)$

`python main.py    --d 3 --T 50000  --num_sim 10  --epsilon 0.01`  

`python main.py  --d 3 --T 50000  --num_sim 10  --epsilon 0.2`

Instances: $\theta=(1,0,0,0,0)$, $\mathcal X=(1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1),(1-\epsilon,2\epsilon,0,0,0),(1-\epsilon,0,2\epsilon,0,0),(1-\epsilon,0,0,2\epsilon,0),(1-\epsilon,0,0,0,2\epsilon)$

`python main.py    --d 5 --T 100000  --num_sim 10  --epsilon 0.01`  

`python main.py    --d 5 --T 100000  --num_sim 10  --epsilon 0.2`  

### research on epsilon

`python main.py    --d 2 --T 10000  --num_sim 10  --epsilon 0.005 --research_on_epsilon 1`

`python main.py    --d 2 --T 10000  --num_sim 10  --epsilon 0.01 --research_on_epsilon 1`

`python main.py    --d 2 --T 10000  --num_sim 10  --epsilon 0.05 --research_on_epsilon 1`

`python main.py    --d 2 --T 10000  --num_sim 10  --epsilon 0.1 --research_on_epsilon 1`

`python main.py    --d 2 --T 10000  --num_sim 10  --epsilon 0.15 --research_on_epsilon 1`

`python main.py    --d 2 --T 10000  --num_sim 10  --epsilon 0.2 --research_on_epsilon 1`

### random examples

`python main.py --seed 1  --K 3 --d 2 --T 50000  --num_sim 10 --verbose`

`python main.py --seed 1 --K 5 --d 3 --T 50000  --num_sim 10 --verbose`

`python main.py --seed 1 --K 9 --d 5 --T 50000  --num_sim 10 --verbose`

`python main.py --seed 1 --K 50 --d 20 --T 50000  --num_sim 10 --verbose`

## Citation
```
@inproceedings{ren2024optimal,
  title={Optimal Batched Linear Bandits},
  author={Ren, Xuanfei and Jin, Tianyuan and Xu, Pan},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
```
