# Energy renovation project for a building

fractional knapsack algorithm for optimizing energy renovation project

## Context

The developement & application of this greedy algorithm have raised as I had the task to suggest restoration measures for buildings at the amperias GmbH company in Germany. The calculation method of the heat load is according to the standard [DIN 12831](https://www.din.de/de/mitwirken/normenausschuesse/nhrs/veroeffentlichungen/wdc-beuth:din21:261292587). In the course of this calculation, for each building structure component, such as walls, roofs, etc., an insulation material has been chosen from a set of choices.

## Objective

An energy consultant assesses the building insulation, sets the requirements, but he/she has many insulation materials for each building component, which responds to the energetic requirements, namely the heating load. But, the performace of each insulation material reduces over time (the rate of depreciation). A well thoughtful choice for each building component should be proactive.
Therefore, I reformulate the objective as to gain the maximum insulation, e.g. minimum heat loss costs, for the whole building while investing the maximum of a given money amount.

## Conditions

- Condition 1: to take into account the rate of depreciation, I set hereafter the lifespan of the building renovation project with 20 years. The assessement considers then heat losses after 20 years of usage for each insulation material.
- Condition 2: the renovation of a building element can be left out, limitation either due to no sufficient investment amount or because the impact of leaving the building element without any renovation is irrelevant. In the code, we can have as a result: (number of renovated elements) counter < (total number of building elements) df.shape[0]

On the other hand, many other factors such as the varying price of the insulation material in the markt, I give with this code an automated solution to possible biases.

## Nomenclature

For explenation purpose, I use both signs [] and {} with combined definitions in logics in Math and the programming language Python.

- Inv: the total renovation investment costs
- Hij: heat loss costs for each insulation material after 20 years of usage for each renovation measure j and building element i (In German: Wärmeverlustkosten)
- Kij: the material costs for each renovation measure j and building element i (Materialkosten)
- n: total number of building elements
- m: the maximum count of measures over all building elements for a normalized problem
- e: the index of the building element in the algorithm
  Here is an explenation:

  Mathematical problem:

  - Set E = {E1, E2, ..., En} and each building element Ei has Mij as a set of its possible renovation measures, for i in [1, ..., n] and j in [1, ..., m]</br>
  - In the example given in the image:</br>

    - n = 4
    - E = {"Window", "Ceiling", "Roof", "Wall"}
    - M1 = {"Single-Glazed Window", "Double glazed window"}
    - M2 = {"Mineral wool", "Spray foam", "Rigid foam boards", "Cellulose", "Sheep wool"}
    - M3 = {"Structural insulated panels", "Foam board", "Rigid foam"}
    - M4 = {"Blanket: batts and rolls", "Concrete blocks insulation", "Insulating concrete forms (ICFs)", "Foam board or rigid foam", "Reflective system"}</br>

  Normalized problem:

  - In the programming code, we normalize the sets Mij for all i values in [1, ..., n], the set Mij could be written Mim and the above mentioned example becomes:
    - n = 4, m = 5
    - E = {"Window", "Ceiling", "Roof", "Wall"}
    - M15 = {"", "", "", "Single-Glazed Window", "Double glazed window"}
    - M25 = {"Mineral wool", "Spray foam", "Rigid foam boards", "Cellulose", "Sheep wool"}
    - M35 = {"", "", "Structural insulated panels", "Foam board", "Rigid foam"}
    - M45 = {"Blanket: batts and rolls", "Concrete blocks insulation", "Insulating concrete forms (ICFs)", "Foam board or rigid foam", "Reflective system"}
  - Note that, the measure "" in each set measures means that no measure could be taken, so the costs are taken in the algorithm 0.0 price unit (price unit: €, $, ... etc.). The costs' sets could be then written for example:
    - K15 = [0.0, 0.0, 0.0, 10, 13]
    - K25 = [45, 56, 67, 18, 19]
    - K35 = [0.0, 0.0, 32.5, 16, 11]
    - K45 = [93, 34, 45, 95, 23]

## Study case

To grafically present the problem, here is an example of possible measures for a set of building components.

![Renovation project - Presentation example with four building components](<resources/renovation for a building - diagram.jpg>)

The deduction of the values of Hij for each insulation material and building component can be done after heat loss calculations with the norm [DIN 12831](https://www.din.de/de/mitwirken/normenausschuesse/nhrs/veroeffentlichungen/wdc-beuth:din21:261292587).

The deduction, as per the picture above, gives the following:

- U-value ================== DIN 12831 ======================> Hij
- Cost for each i building element and j renovation material = Kij

**Analogy: fractional knapsack algorithm**

To explain the resolution of the problem, I refer to the basics of the fractional knapsack algorithm in the following [Webpage](https://algodaily.com/lessons/getting-to-know-greedy-algorithms-through-examples/fractional-knapsack-problem). The programmed solution calculates value x weight for each building element, which is in my case the product "heat loss costs" times "material costs" (Kej).

![Example of a fractional knapsack problem](<resources/Fractional Knapsack Problem.png>)

Hereafter I used 4 main steps.

**Package import**

```python
import random
import numpy as np
import pandas as pd
```

**Inputs**
Here you can change the values for investment costs, number of building elements and the m number, which is defined for the normalized problem as the maximum count of measures over all building elements.

```python
# The total renovation investment costs
Inv = 500
# the total number of building elements
n = 4
# the maximum count of measures over all building elements for a normalized problem
m = 5
```

**Parameters of the measures**
Values related to material costs and heat loss costs after 20 years of usage are set randomly for experimental purposes. Their values can also be loaded from other data sources, such as CSV-files.

**Experimental runs**
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

#### Step 1: create a Pandas dataframe

As mentioned in the Condition 2 in README.md file, the renovation of a building element can be left out, limitation either due to no sufficient investment amount or because the impact of leaving the building element without any renovation is irrelevant.

So, we should add 0.0 price unit to the dictionary d and consequently to the dataframe df.

```python
for i in range(len(d["Kij"])):
    d["Kij"][i] = np.insert(d["Kij"][i], [0], [0], axis=0)
df = pd.DataFrame(data=d)
```

**Diplay the dataframe df**

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

#### Step 2: loop over the set of building elements

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

#### Step 3: sort & reorder the building elements

```python
# sort ascendingly
Min.sort(key=lambda x: x[1])
dt = pd.DataFrame(data=Min)
dt.columns = ["e", "Min_e"]
```

**Diplay the dataframe dt**

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

#### Step 4: Investment plan

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

## Credits

Credits goes to [algodaily](https://algodaily.com/) for their lessons' webpage in fractional knapsack problem.
