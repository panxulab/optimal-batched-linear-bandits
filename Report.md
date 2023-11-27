## Report

### Random Example

Generate random instances with different arm numbers and context dimensions.

`python main.py --seed 1  --K 3 --d 2 --T 50000  --num_sim 10 --verbose`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125172914953.png" alt="image-20231125172914953" style="zoom:50%;" />

> Batch complexity:
> E3TC: mean: 3.000000, std: 0.000000
> Phased elimination with D-optimal exploration: mean: 3.100000, std: 0.300000
> OFUL: mean: 39.000000, std: 0.000000

`python main.py --seed 1 --K 5 --d 3 --T 50000  --num_sim 10 --verbose`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125173012315.png" alt="image-20231125173012315" style="zoom:50%;" />

> Batch complexity:
> E3TC: mean: 3.000000, std: 0.000000
> Phased elimination with D-optimal exploration: mean: 3.900000, std: 0.300000
> OFUL: mean: 61.800000, std: 0.400000

`python main.py --seed 1 --K 9 --d 5 --T 50000  --num_sim 10 --verbose`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125173138389.png" alt="image-20231125173138389" style="zoom:50%;" />

> Batch complexity:
> E3TC: mean: 3.000000, std: 0.000000
> Phased elimination with D-optimal exploration: mean: 4.000000, std: 0.000000
> OFUL: mean: 99.900000, std: 0.700000

`python main.py --seed 1 --K 50 --d 20 --T 100000  --num_sim 20 --verbose`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125173705393.png" alt="image-20231125173705393" style="zoom:50%;" />

>Batch complexity:
>E3TC: mean: 3.000000, std: 0.000000
>Phased elimination with D-optimal exploration: mean: 4.000000, std: 0.000000
>OFUL: mean: 388.450000, std: 0.589491

### The end of optimism instances

Instances where algorithms based on optimism proved to be sub-optimal.

$\epsilon=0.01,0.2$

Instances: $\theta=(1,0)$, $\mathcal X=(1,0),(1-\epsilon,2\epsilon),(0,1)$

`python main.py   --seed 1 --d 2 --T 10000  --num_sim 10  --epsilon 0.01`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125172132655.png" alt="image-20231125172132655" style="zoom: 50%;" />

> Batch complexity:
> E3TC: mean: 3.000000, std: 0.000000
> Phased elimination with D-optimal exploration: mean: 4.000000, std: 0.000000
> OFUL: mean: 37.000000, std: 0.000000

`python main.py   --seed 1 --d 2 --T 10000  --num_sim 10  --epsilon 0.2`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125172330643.png" alt="image-20231125172330643" style="zoom:50%;" />

> Batch complexity:
> E3TC: mean: 3.000000, std: 0.000000
> Phased elimination with D-optimal exploration: mean: 4.000000, std: 0.000000
> OFUL: mean: 37.000000, std: 0.000000

Instances: $\theta=(1,0,0)$, $\mathcal X=(1,0,0),(0,1,0),(0,0,1),(1-\epsilon,2\epsilon,0),(1-\epsilon,0,2\epsilon)$

`python main.py   --seed 1 --d 3 --T 30000  --num_sim 10  --epsilon 0.01`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125172426802.png" alt="image-20231125172426802" style="zoom:50%;" />

> Batch complexity:
> E3TC: mean: 3.000000, std: 0.000000
> Phased elimination with D-optimal exploration: mean: 4.000000, std: 0.000000
> OFUL: mean: 59.100000, std: 0.538516

`python main.py   --seed 1 --d 3 --T 30000  --num_sim 10  --epsilon 0.2`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125172521543.png" alt="image-20231125172521543" style="zoom:50%;" />

> Batch complexity:
> E3TC: mean: 3.000000, std: 0.000000
> Phased elimination with D-optimal exploration: mean: 4.000000, std: 0.000000
> OFUL: mean: 59.400000, std: 0.489898

Instances: $\theta=(1,0,0,0,0)$, $\mathcal X=(1,0,0,0,0),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1),(1-\epsilon,2\epsilon,0,0,0),(1-\epsilon,0,2\epsilon,0,0),(1-\epsilon,0,0,2\epsilon,0),(1-\epsilon,0,0,0,2\epsilon)$

`python main.py   --seed 1 --d 5 --T 50000  --num_sim 10  --epsilon 0.01`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125172633876.png" alt="image-20231125172633876" style="zoom:50%;" />

> Batch complexity:
> E3TC: mean: 3.000000, std: 0.000000
> Phased elimination with D-optimal exploration: mean: 4.000000, std: 0.000000
> OFUL: mean: 98.700000, std: 0.640312

`python main.py   --seed 1 --d 5 --T 50000  --num_sim 10  --epsilon 0.2`

<img src="C:\Users\l\AppData\Roaming\Typora\typora-user-images\image-20231125172735042.png" alt="image-20231125172735042" style="zoom:50%;" />

> Batch complexity:
> E3TC: mean: 3.000000, std: 0.000000
> Phased elimination with D-optimal exploration: mean: 4.000000, std: 0.000000
> OFUL: mean: 98.400000, std: 0.489898
