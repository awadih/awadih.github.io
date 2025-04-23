# Study case: renovation project for a building
we apply here the algorithm in 'algorithm.ipynb' to our study case in the picture (see README.md).

#### Package import


```python
import random
import numpy as np
import pandas as pd
```

#### Inputs
Here you can change the values for investment costs, number of building elements and the m number, which is defined for the normalized problem as the maximum count of measures over all building elements.


```python
# The total renovation investment costs
Inv = 5000
# the total number of building elements
n = 4
# the maximum count of measures over all building elements for a normalized problem 
m = 5
```

#### Parameters of the measures 
Values related to material costs and heat loss costs after 20 years of usage are set randomly for experimental purposes. Their values can also be loaded from other data sources, such as CSV-files.

##### Experimental runs 
As no real values are available, I use hereafter experimental, sampled values:
- Material costs are taken with 1 decimal point and generated using samples from a uniform distribution from low value of 50 to the highest value of 300 price unit
- Heat loss costs after 20 years of usage from 10 to 1000 units


```python
d = {"e": [], "Kij": [], "Hij": []}
for i in range(n):
    d["e"].append(i)
    # Material costs
    d["Kij"].append(np.round(np.random.uniform(low=50, high=300, size=(m-1,)), decimals=1))
    # Heat loss costs TODO("Rethink low and high values")
    d["Hij"].append(np.round(np.random.uniform(low=10, high=1000, size=(m,)), decimals=1))
```

#### Step 1:

As mentioned in the Condition 2 in README.md file, the renovation of a building element can be left out, limitation either due to no sufficient investment amount or because the impact of leaving the building element without any renovation is irrelevant.

So, we should add 0.0 price unit to the dictionary d and consequently to the dataframe df.


```python
for i in range(len(d["Kij"])):
    d["Kij"][i] = np.insert(d["Kij"][i], [0], [0], axis=0)
df = pd.DataFrame(data=d)
```

##### Diplay the dataframe df


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>e</th>
      <th>Kij</th>
      <th>Hij</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>[0.0, 150.7, 272.4, 294.1, 92.9]</td>
      <td>[408.2, 33.3, 921.1, 270.5, 666.7]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>[0.0, 98.8, 157.9, 284.4, 256.1]</td>
      <td>[410.5, 45.0, 97.1, 419.4, 472.6]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>[0.0, 160.0, 280.5, 219.4, 153.2]</td>
      <td>[964.4, 148.8, 518.1, 748.6, 139.6]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>[0.0, 68.0, 294.2, 107.6, 81.2]</td>
      <td>[293.3, 530.7, 618.7, 332.8, 842.2]</td>
    </tr>
  </tbody>
</table>
</div>



#### Step 2: towards the fractional knapsack problem

Find the indices in the dataframe of the greedy renovation measures. 

To do so, we calculate $\min_{e}$ for each e building element. We define hereafter some helpers variables and find sort their values.


```python
indices = []
helper = []
Min = []
for i in df.index:
    product = [km * kg for (km, kg) in zip(df["Kij"][i], df["Hij"][i])]
    indices.append((i, product.index(min([value for value in product if value > 0]))))
    helper.append(i)
for i in df.index:
    k = helper.index(i)
    Min.append((i, np.round(df["Kij"][i][indices[k][1]] * df["Hij"][i][indices[k][1]], decimals=1)))
```

#### Step 3: sort & reorder


```python
# sort ascendingly
Min.sort(key=lambda x: x[1])
dt = pd.DataFrame(data=Min)
dt.columns = ["e", "Min_e"]
```

##### Diplay the dataframe dt

Note that the elements are well ascendingly sorted.


```python
dt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>e</th>
      <th>Min_e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>4446.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>5018.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>21386.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>35809.3</td>
    </tr>
  </tbody>
</table>
</div>



#### Step 4:
Invest Inv starting with the lowest value of $\min_{e}$ with some help-variables


```python
# Display for the project's inputs:
print("Renovation project's infos:")
print("\tThe number of building elements:\t{}.".format(n))
print("\tThe total renovation investment costs:\t{} price unit.".format(Inv))
# Display for the investement plan:
print("\nInvestement plan with ordered measures:")
count = 0
toInvest = Inv
sum = 0
counter = 0
while toInvest > 0 and count < len(Min):
    flag = helper.index(Min[count][0])
    if toInvest > df["Kij"][Min[count][0]][indices[flag][1]]:
        toInvest -= df["Kij"][Min[count][0]][indices[flag][1]]
        print(
            "\tRenovation measure {}\tKij = {} price unit.\t\tHij = {} price unit.".format(Min[count][0], df["Kij"][Min[count][0]][indices[flag][1]],
                                     df["Hij"][Min[count][0]][indices[flag][1]]))
        sum += df["Kij"][Min[count][0]][indices[flag][1]]
        counter += 1
        count += 1
    else:
        pass
print("\n\tNumber of chosen buiding elements for renovation: {}.".format(counter))
print("\tInvested sum: {} price unit.".format(sum))
```

    Renovation project's infos:
    	The number of building elements:	4.
    	The total renovation investment costs:	5000 price unit.
    
    Investement plan with ordered measures:
    	Renovation measure 1	Kij = 98.8 price unit.		Hij = 45.0 price unit.
    	Renovation measure 0	Kij = 150.7 price unit.		Hij = 33.3 price unit.
    	Renovation measure 2	Kij = 153.2 price unit.		Hij = 139.6 price unit.
    	Renovation measure 3	Kij = 107.6 price unit.		Hij = 332.8 price unit.
    
    	Number of chosen buiding elements for renovation: 4.
    	Invested sum: 510.29999999999995 price unit.

