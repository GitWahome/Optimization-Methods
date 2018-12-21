

```python
import numpy as np
import cvxpy as cvx
from timeit import default_timer as time
import collections
```

Background:

My Aunt operates a trucking logistics firm in Dallas Texas.  They key to this operation is the dispatch process. 
I analyzed  how they run it and realized that they make use of a greedy approach. They receive orders from a listings board, call the list poster and negotiate a price.
There are several factors that determine whether they take a specific order. Among them are the size of the objects to be moved, the weight of the objects listed, the destination of the order, the expected roadtime which can be infered from the delivery date among others.

This in essence presents a real time Knapsack problem but there are some nuances that make it a bit of a challenge but also allow for some assumptions.

**PROBLEM (S)**:
 
 There are a number of subproblems that this set up presents all of which  can be solved using various convex and non convex approaches.

My approach to this has been to define a subproblem and then solve it using CVCPY, each time justifying why I thought he approach taken fits the context.

For a start, we will define some constants which we must take into account. 


*   A logistics company can lease trucks of any size.. 

*   Heavy Class 7 trucks have a carrying capacity of 55000 pounds.

*   Small class 2 trucks have a carrying capacity of  26000 pounds.

*   Medium class 6 trucks have a carrying capacity of 33000 pounds.


*(I'm so sorry for using pounds, but you know, the backward Americans)*

The objective is simple,  they  need to minimize the number of trucks while maximizing the weight carried by each container.


One catch is, how many goods are put in one container is dependent only on the weight of the content already in the box weight. The profits though are mainly determined by the weight of the goods being transfered hence a carrying weight capacity constraint is imposed depending on the trucks available. 

**This makes evaluation easy as we only need to track the weight of the boxes when evaluating profits. The problem thus lies in optimizing for volume while maximizing profit. I make the assumption that all vehicles are available and if not, then they can always lease a new vehicle to do the work This is where my main challenge is.**

Transport  costs are dependent on the  weight and distance that the goods need to travel. The profit is entirely dependent on the weight and distance. To extend our objective function,  **We can determine the expected profit for each item. The objective is to maximize the weight and carried by each container given the volume and weight of each item. **

NOTE: I acknowledge the essence of route optimization but that introduces a layer of complexity hence I am abstracting the routes. 

I do not have the exact number but let us assume:

1.   The maximum consumption of a heavy truck is say 55.6 per 100 Kilometers  and the minimum is 18 when it is not carrying anything(Large trucks),

2.  The maximum consumption of a medium truck is say 40.5 liters per Kilometer  and the minimum is 10 when it is not carrying anything(Medium trucks)

3.   The maximum consumption of a small truck is say 34 liters per Kilometer  and the minimum is 5 when it is not carrying anything(Small trucks)

The weight relative to the maximum carrying capacity will determine the fuel consumption. We can evaluate the expected profits by multiplying the expected fuel consumption by the cost of fuel which we i assume to be 10$ per liter. 

I set a weight check too to be returned  if the load weight is within the vehicle limit. This allows us to know which car to use since we dont want to go over the limit to avoid getting citations.

Any car can travel any distance so there are no distance constraints, if anything, distance is good since we will make more money We just have to monitor how the fuel is affected by the weight.
(In real life however, repair costs make it such that it is more efficient to use large trucks over long distances. I abstract this to keep things relatively simple)




```python
def fuel_cost(weight, distance, truck_class, weight_check = True, fuel_rate=10):
  if truck_class == "H":
    max_weight, min_consumption, max_consumption= 55000, 18, 55.6
    if weight>max_weight:
        weight_check = False
  if truck_class == "M":
    max_weight, min_consumption, max_consumption= 33000, 10, 40.5
    if weight>max_weight:
      weight_check = False
  if truck_class == "S":
    max_weight, min_consumption, max_consumption= 26000, 5, 34
    if weight>max_weight:
      weight_check = False
  return (min_consumption+(weight/max_weight)*(max_consumption-min_consumption))* fuel_rate *(distance/100), weight_check

print("1KM - MAX SMALL CAPACITY: 26000")
print("MEDIUM",fuel_cost(26000, 1,"M"))
print("SMALL",fuel_cost(26000, 1,"S"))
print("\nSLIGHTLY HIGHER: 26500")
print("MEDIUM",fuel_cost(26500, 1,"M"))
print("SMALL",fuel_cost(26500, 1,"S"))

print("\n_______________________________________________________________________")


print("1KM -MAX MEDIUM CAPACITY: 26000")
print("HEAVY",fuel_cost(33000, 1,"H"))
print("MEDIUM",fuel_cost(33000, 1,"M"))
print("\nSLIGHTLY HIGHER: 28500")
print("HEAVY",fuel_cost(33500, 1,"H"))
print("MEDIUM",fuel_cost(33500,1, "M"))

```

    1KM - MAX SMALL CAPACITY: 26000
    MEDIUM (3.4030303030303033, True)
    SMALL (3.4, True)
    
    SLIGHTLY HIGHER: 26500
    MEDIUM (3.449242424242424, True)
    SMALL (3.455769230769231, False)
    
    _______________________________________________________________________
    1KM -MAX MEDIUM CAPACITY: 26000
    HEAVY (4.056, True)
    MEDIUM (4.05, True)
    
    SLIGHTLY HIGHER: 28500
    HEAVY (4.090181818181819, True)
    MEDIUM (4.096212121212122, False)
    

We observe that smaller trucks are more efficient when the load is smaller. Large trucks with their larger capacity are more efficient for heavy loads. Any car can carry over Its capacity but then the consumption gets larger.

I have based my limits on true data but adjusted the range of consumption such that each vehicle is the most efficient at Its maximum capacity(And still is going slightly higher) but the next vehicle class becomes more efficient quickly and surpases it. 

In real life, going over the capacity entails more repair/maintenance costs thus it always makes sense to use the next vehicle class if the capacity is larger than the prescribed limit.

Details on fuel consumption by weights are provided [in this wikipedia article](https://en.wikipedia.org/wiki/Fuel_efficiency)

This sets us the problem quite nicely.  Once we receive orders which entails distances and weights, and volume, we simply have to analyze the expected profits from each item. I assume the charge is 2USD per KM. The Cost depends on the weight too. I will assume a uniform charge of 2.5 USD per lb. This thus gives us a charges function for any given load as: 2distance +2.5load. It should be higher than the fuel cost otherwise the whole operation is a waste. These values are such that we still make a profit even if we are transporting a 1 Kilo load over a distance of 1 kilometer.

Normally the price is negotiated an agreed upon thus if a customer is a noob, they can overpay but in some instances, they negotiate too much. For simpicity, I will assume an average cost. 

Quick note: In real like, we can have a minimum average and during negotiation, all we need to do is ensure the price paid by a customer does not reduce our average to below the minimum. 

So with 2$/KM in cost, we can evaluate the profit made from item trip by:


```python
chargesF = lambda weight, distance: 2*weight+2.5*distance
print("SMALL",fuel_cost(1, 1,"S"))
print("MEDIUM",fuel_cost(1, 1,"M"))
print("LARGE",fuel_cost(1, 1,"H"))
print(chargesF(1,1))
```

    SMALL (0.5001115384615384, True)
    MEDIUM (1.0000924242424243, True)
    LARGE (1.8000683636363635, True)
    4.5
    


```python
def profits(weights, distances):
  costs_heavy = [fuel_cost(weights[i], distances[i],"H") for i in range(len(weights))]
  costs_medium = [fuel_cost(weights[i], distances[i],"M") for i in range(len(weights))]
  costs_small = [fuel_cost(weights[i], distances[i],"S") for i in range(len(weights))]
  charges = [chargesF(weights[i],distances[i]) for i in range(len(weights))]
  
  return({"H":[(charges[i]-costs_heavy[i][0],costs_heavy[i][1]) for i in range(len(weights))], 
          "M":[(charges[i]-costs_medium[i][0],costs_medium[i][1]) for i in range(len(weights))], 
          "S":[(charges[i]-costs_small[i][0], costs_small[i][1]) for i in range(len(weights))]})
```

Example:


```python
volumes = [850, 400, 150, 200, 550, 220, 1000, 400]
weights = [50000, 5000.25, 2000, 50000, 30000, 40000, 44000.75, 46000]
distances = [500, 250, 1000, 570, 390, 485, 700, 46]
profits(weights, volumes)
```




    {'H': [(97689.54545454546, True),
      (10143.765890909091, True),
      (4084.490909090909, True),
      (99456.36363636363, True),
      (59257.0, True),
      (79552.4, True),
      (85693.44872727273, True),
      (91022.10909090909, True)],
     'M': [(97346.9696969697, False),
      (10415.642272727273, True),
      (4197.272727272727, True),
      (99375.75757575757, False),
      (59300.0, True),
      (79516.66666666667, False),
      (85434.76401515151, False),
      (90899.39393939394, False)],
     'S': [(96959.61538461539, False),
      (10577.411923076923, True),
      (4266.538461538462, True),
      (99284.61538461539, False),
      (59259.61538461538, False),
      (79458.46153846153, False),
      (85093.72403846154, False),
      (90747.69230769231, False)]}



Now let us define a a solve
This should take in the item weights and volumes then give us the minimum number of containers we will need. I have included a solution decomposer that will tell us the optimum content of each container. 

Note: Sometimes weight and volume are not the only factor at play. A client may be transporting radioactive elements, or food in which case, these can only be transported together or wholly alone even if space is wasted. We skip  such cases to keep things simple. These are also easy to handle. We just assign a container to the isolated items and in foods and such, we just do the optimization for orders that are of similar type.

For now, let us solve this generally.


```python
print(cvx.installed_solvers())
box_vol = 2400
box_weight = 55000

def m_solver(factor_1, factor_2, max_capacity_r1, max_capacity_r2, solver=cvx.GLPK_MI):

  max_n_boxes = len(factor_1) # real-world: heuristic?

  """ Optimization """

  # VARIABLES
  trucks_used = cvx.Variable(max_n_boxes, boolean=True)
  truck_volume = cvx.Variable(max_n_boxes)
  truck_weight = cvx.Variable(max_n_boxes)
  truck_item_map = cvx.Variable((max_n_boxes, len(factor_1)), boolean=True)

  # CONSTRAINTS
  cons = []

  # Each item is Transported once
  cons.append(cvx.sum(truck_item_map, axis=0) == 1)
  # Truck is used when >=1 item is using it
  cons.append(trucks_used * (len(factor_1) + 1) >= cvx.sum(truck_item_map, axis=1))
  # Truck vol constraints
  cons.append(truck_item_map * factor_1 <= max_capacity_r1)

  # Truck weight constraints
  cons.append(truck_item_map * factor_2 <= max_capacity_r2)

  problem = cvx.Problem(cvx.Minimize(cvx.sum(trucks_used)), cons)
  start_t = time()
  problem.solve(solver=solver, verbose=True)
  end_t = time()
  print(f'OPTIMUM SOLUTION EVALUATED IN {end_t - start_t} SECONDS')
  print(problem.status)
  print(problem.value)
  print(truck_item_map.value)

  """ Reconstruct solution """
  n_boxes_used = int(np.round(problem.value))
  box_inds_used = np.where(np.isclose(trucks_used.value, 1.0))[0]
  truck_loads = {}
  for truck in range(n_boxes_used):
      truck_loads[truck]=[]
      raw = truck_item_map[box_inds_used[truck]]
      items = np.where(np.isclose(raw.value, 1.0))[0]
      vol_used = 0
      weight_used = 0
      for item in items:
        truck_loads[truck].append((item, (factor_1[item], factor_2[item])))
        vol_used += factor_1[item]
        weight_used += factor_2[item]
  return {"Minimum Containers":n_boxes_used,"Load List":truck_loads,"(Total Volume, Total Weight)":(vol_used, weight_used)}
solution = m_solver(factor_1= volumes,
         factor_2= weights,
         max_capacity_r1= box_vol,
         max_capacity_r2= box_weight)

print(solution)
```

    ['ECOS', 'ECOS_BB', 'CVXOPT', 'GLPK', 'GLPK_MI', 'SCS', 'OSQP']
    OPTIMUM SOLUTION EVALUATED IN 4.0120983 SECONDS
    optimal
    6.0
    [[0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 1. 0. 1. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]]
    {'Minimum Containers': 6, 'Load List': {0: [(5, (220, 40000))], 1: [(7, (400, 46000))], 2: [(1, (400, 5000.25)), (2, (150, 2000)), (4, (550, 30000))], 3: [(0, (850, 50000))], 4: [(3, (200, 50000))], 5: [(6, (1000, 44000.75))]}, '(Total Volume, Total Weight)': (1000, 44000.75)}
    

I now need a helper function to make sense of the data.
NOTE: I have incorporated this into my profit evaluator above.  

One may wonder how we optimize for container size. The set up is such that if the load exceeds the maximum carrying capacity of the vehicle container then the whole loading is disqualified as the weightchecker is set to 0. I am assuming that all the containers are closed but in real life,we can have open containers. Once  the system has optimized the load for volume and weight, we can simply look at these numbers and determine which vehicles to lease. This is merely a matter of tracing back all instances where the weight check is passed and the profits for the instances where it has passed. We pick the vehicle class that gives us maximum profit. 

For each container, we evaluate the expected profit for the three vehicle  classes, picking the vehicle class that gives us maximum profit.

We do this then output the total profit and the leased vehicles recommendation.


```python
truck_classes = {0:"HEAVY",1:"MEDIUM",2:"SMALL",}
def valid_profs(items, distances):
    #print(items)
    profs = []
    p_main = []
    for it in items:
        v,w =it[1]
        p_set = []
        for p in profits([w],[distances[solution_processor.count]]).values():
            #Check if weight check is passed
            if p[0][1]:
                p_set.append(p[0][0])
            else:
                #Set a big -ve if weight check is not passed
                p_set.append(-1000000)
        p_main.append(p_set)

        solution_processor.count+=1
    return [sum(x) for x in zip(*p_main)] #Sum all valid profits to see which system has the largest total. 

def solution_processor(solution, distances, item_names="Empty"):
    solution_processor.count = 0
    expected_profit, expected_class = [], []
    truck_class ={0:"HEAVY", 1:"MID", 2:"SMALL"}
    for truck in range(len(solution['Load List'])):
        print(f"\nDISPATCHING TRUCK NUMBER: {truck+1}")
        items = solution['Load List'][truck]
        profits = valid_profs(items, distances)
        print(f"TRUCK SIZE: {truck_classes[profits.index(max(profits))]}")
        if item_names is "Empty":
            print("Items List")
            for it in items:
                print(f"Item {it[0]} with volume {it[1][0]} ft3 and weight {it[1][1]} lbs")
        else:
            print("Items List")
            for it in items:
                print(f"Item {item_names[it[0]]} with weight {it[1][0]} ft3  and volume {it[1][1]}lbs")
        print(f"Maximum Profit is about {int(max(profits))} $")
optimum_vehicles = solution_processor(solution, distances)

```

    
    DISPATCHING TRUCK NUMBER: 1
    TRUCK SIZE: HEAVY
    Items List
    Item 5 with volume 220 ft3 and weight 40000 lbs
    Maximum Profit is about 78982 $
    
    DISPATCHING TRUCK NUMBER: 2
    TRUCK SIZE: HEAVY
    Items List
    Item 7 with volume 400 ft3 and weight 46000 lbs
    Maximum Profit is about 91388 $
    
    DISPATCHING TRUCK NUMBER: 3
    TRUCK SIZE: MEDIUM
    Items List
    Item 1 with volume 400 ft3 and weight 5000.25 lbs
    Item 2 with volume 150 ft3 and weight 2000 lbs
    Item 4 with volume 550 ft3 and weight 30000 lbs
    Maximum Profit is about 75291 $
    
    DISPATCHING TRUCK NUMBER: 4
    TRUCK SIZE: HEAVY
    Items List
    Item 0 with volume 850 ft3 and weight 50000 lbs
    Maximum Profit is about 98681 $
    
    DISPATCHING TRUCK NUMBER: 5
    TRUCK SIZE: HEAVY
    Items List
    Item 3 with volume 200 ft3 and weight 50000 lbs
    Maximum Profit is about 98097 $
    
    DISPATCHING TRUCK NUMBER: 6
    TRUCK SIZE: HEAVY
    Items List
    Item 6 with volume 1000 ft3 and weight 44000.75 lbs
    Maximum Profit is about 87895 $
    

Now let us see how this system works with some real life data.

It should be noted: No container car transport goods greater than the maximum weight capacity of 55000. I would ideally filter such cargo out but in my system, I will not include them entirely. 

The largest closed shipping container is 40 ft long: http://seaplus.com/container.html
It has a capacity of 2393 cubic ft and a payload of 58600lbs. This container should be able to carry any load. The size of the container will depend on the volume of the items the system is assigned. Our output is such that one needs to inspect, see the total volume of the items and determine which  container works best. They can always use a bigger one either way.

Now that we have a functional optimizer, let us work on the 'real' data.


```python
import pandas as pd
import numpy as np
Names = ["New Computers", "Hardware supplies", "Concrete", "Building utilities", "Industry tools", 
         "Drill bits", "Donation clothes", "Power tools"]

Volumes = [np.random.randint(5,2400) for i in range(len(Names))]
Weights = [np.random.randint(5,55000) for i in range(len(Names))]
Distances = [np.random.randint(5,5500) for i in range(len(Names))]

data = {
    'Item Name':Names,
    'Distance':Distances,
    'Volume':Volumes,
    'Weight':Weights,
}
data =pd.DataFrame(data)
data.head()
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
      <th>Item Name</th>
      <th>Distance</th>
      <th>Volume</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New Computers</td>
      <td>5171</td>
      <td>1225</td>
      <td>17076</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hardware supplies</td>
      <td>1435</td>
      <td>1217</td>
      <td>16195</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Concrete</td>
      <td>889</td>
      <td>1065</td>
      <td>47438</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Building utilities</td>
      <td>3072</td>
      <td>2376</td>
      <td>34625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Industry tools</td>
      <td>472</td>
      <td>1501</td>
      <td>34446</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(len(data['Volume']))
```

    8
    

Now lets run the solve on this. I generated the data so apologies for the senseless Item names. Computers are dumb.


```python
solution = m_solver(factor_1= list(data['Volume'])[:10],
         factor_2= list(data['Weight'])[:10],
         max_capacity_r1= box_vol,
         max_capacity_r2= box_weight)
```

    OPTIMUM SOLUTION EVALUATED IN 48.1055118 SECONDS
    optimal
    7.0
    [[0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1.]]
    


```python
optimum_vehicles = solution_processor(solution,data['Distance'], data['Item Name'] )
```

    
    DISPATCHING TRUCK NUMBER: 1
    TRUCK SIZE: SMALL
    Items List
    Item Drill bits with weight 1676 ft3  and volume 1194lbs
    Maximum Profit is about 12041 $
    
    DISPATCHING TRUCK NUMBER: 2
    TRUCK SIZE: HEAVY
    Items List
    Item Building utilities with weight 2376 ft3  and volume 34625lbs
    Maximum Profit is about 66857 $
    
    DISPATCHING TRUCK NUMBER: 3
    TRUCK SIZE: SMALL
    Items List
    Item Hardware supplies with weight 1217 ft3  and volume 16195lbs
    Maximum Profit is about 32562 $
    
    DISPATCHING TRUCK NUMBER: 4
    TRUCK SIZE: SMALL
    Items List
    Item New Computers with weight 1225 ft3  and volume 17076lbs
    Item Donation clothes with weight 871 ft3  and volume 17933lbs
    Maximum Profit is about 70310 $
    
    DISPATCHING TRUCK NUMBER: 5
    TRUCK SIZE: HEAVY
    Items List
    Item Industry tools with weight 1501 ft3  and volume 34446lbs
    Maximum Profit is about 64197 $
    
    DISPATCHING TRUCK NUMBER: 6
    TRUCK SIZE: HEAVY
    Items List
    Item Concrete with weight 1065 ft3  and volume 47438lbs
    Maximum Profit is about 83773 $
    
    DISPATCHING TRUCK NUMBER: 7
    TRUCK SIZE: SMALL
    Items List
    Item Power tools with weight 1741 ft3  and volume 6192lbs
    Maximum Profit is about 12649 $
    

And just like that, we have an optimizer for a real world problem.

# TASKS

## Problem Description: 

This is an extended Knapsack problem that was aimede at optimizing the carrying combinations for each truck in a transporation logistics firm. 

It is a mixed integer programming question given the nature of the variables it takes. This is because the constrains imposed on the weights and volumes are affine but the system can optimize for non integer problems.It is honestly a dual system where if you pass in only integers, it will solve for them, but it does not mind mixed integer problems which is the case for our first example.  The good power of GLPK_MI.

It makes use of the brench and bound method.
The nature of our feasible set is not convex. This is because the constraints imposed by the volume and weights are intermittent over the feasible region. We we to think of the region imposed by a convex problem with similar constraints, if we intrduced the weight and volume constraints, they would form 'stripes' of infeasible sections within the same which cut off interconnectivity within some points of the region thus region will not be convex. This is thus not a convex problem.

I have included comments that detail what I did in the code, where variables were created, constraints were defined among others.

## Analysis.

I found plots not to be necessary for this but I havee included outputs that give an intuition of what is expected of each function. I have also included captions that explain the purpose of each.

# References:

https://towardsdatascience.com/integer-programming-in-python-1cbdfa240df2
https://stackoverflow.com/questions/48866506/binpacking-multiple-constraints-weightvolume


# Appendix

None

# Extension.

These arent hard honestly, but I decided to take on simple, Linear programming problem just to span into the convex region.

I love PatrickJMT, one of those old tutorial makes who are actually better than Khan Academy.
This is his tutorial on the graphical way of solving Linear Programming problems. I implemented solutions to all of these in CVXOPT.


# Problem 1:

https://www.youtube.com/watch?v=M4K6HYLHREQ

240 Acres of land.
Profit: $40/acre corn, $40/acre oats.
Have 320 Hours Availablle.
Corn Takes 2 Hours of labor per acre, oats requires 1 hour of labor to be planted.
How many acres should each be planted to maximize profits?

## Formulation.

x= Acres of corn.
y = Acres of oats.

Objective: Maximize: p =40x+30y.
Constraints: x>=0, y>=0 (Something must be planted)
x+y<=240. (Hours)
2x+y<=320. (Labor)



```python
import cvxpy as cp

# Our variables.
x = cp.Variable()
y = cp.Variable()

# Create two constraints.
constraints = [x>=0, y>=0, x + y <= 240, 2*x + y <= 320]

# Form objective.
obj = cp.Maximize(40*x + 30*y)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print(f"optimal {x.value} corn and {y.value} oats")
```

    status: optimal
    optimal value 7999.999999999999
    optimal 79.99999999999997 corn and 160.00000000000003 oats
    

What I liked about this examples is that we saw some errors in Patricks evaluation which he later fixed.

This is a classic convex programming question. 
The nature of the feasible region is convex. This can be seen graphically in the video and also intuitively, all the variables are continuous and the whole point is to optimize for them. In fact, the simplex method involves mapping the objective funvtion to the maximum or minimum vertice to find the maxima or minima. The optimal solution in this kind of problem will always be on a vertex.

# Problem 2

https://www.youtube.com/watch?v=I7brzjFhYEU

Its pretty much a similar problem but is a minimization problem. This is perhaps even simple. I can make use of the same system above but just negating the objective in order to maximize in lieu of minimizing but I will just use minimize anyway.

Set up:

Minimize C=4x+2y ST

-3x+2y<=6

3x+y<=3

y>=0



```python
import cvxpy as cp

# Our variables.
x = cp.Variable()
y = cp.Variable()

# Create two constraints.
constraints = [y>=0, -3*x + 2*y <= 6, 3*x + y <= 3]

# Form objective.
obj = cp.Minimize(4*x + 2*y)

# Form and solve problem.
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value)
print(f"optimal {x.value} for x and {y.value} for y")
```

    status: optimal
    optimal value -8.0
    optimal -2.0 for x and 3.256467207001122e-30 for y
    

This was a lovely one. It only goes to show the iterative nature of cvx py solvers. The y value is 0 but cvxpy gave us a very small number. It does not harm, so long as it works. We basically replicated the whole tutorial set by Patrick using a modern optimizer. Goodbye to the paper and markpen he used however lovely thy are to look at. 
