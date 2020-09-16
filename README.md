
# Sampling and the Central Limit Theorem

![sample](https://media.giphy.com/media/OsOP6zRwxrnji/giphy.gif)

# Agenda 

1. Differentiate terms: discriptive/inferential, statistics population/sample, paramater/statistic, sample distribution/sampling distribution
2. Define and calculate standard error
3. Use Numpy to randomly sample a distribution
4. Describe the central limit theorem and connect it to our knowledge of distributions and sampling.
5. Divy Example Continued

## Probability vs Statistics
- Probability starts with known probabilities and obtains how probable any particular observation would be
- Statistics works the other way around. Start with and observations (data) and try to determine its probability

## Descriptive vs Inferential Statistics
- Descriptive Statistics
   > simply describe what is observed. The average height of a high school football team can be directly calculated by measuring all of the current players height.
- Inferential statistics 
    > try to say something general about a larger group of subjects than those we have measured. For example, we would be doing inferential statistics if we wanted to know about the average height of all high school football teams.
    - To put it another way, statistical inference is the process by which we take observations of a subset of a group and generalize to the whole group.

## Population Inference

The mayor's office has hired Flatiron Data Science Immersive students to determine a way to fix traffic congestion. A good starting point is to determine what proportion of the population of Seattle owns a car.

![traffic](https://media.giphy.com/media/3orieWY8RCodjD4qqs/giphy.gif)

In order for us to make any determinations about a population, we must first get information about it.

Because it's usually completely impractical to get data about *everyone* in a population, we must take a sample.

## Key Terms
 - the entire group is known as the **population**  
 - the subset is a known as the **sample**


![pop](./img/sample_pop.png)

- We would use samples if the population is:
    - Too big to enumerate
    - too difficult/time consuming or expensive to sample in its entirety.

**Random sampling is not easy to do**  
Continuing our Seattle car example, how would we take a sample? 

Here are two strategies we might employ:

* Stand outside of Flatiron at 12 pm and ask random people until *n* responses


* Go to a randomly assigned street corner and at a random time and ask *n* people if they own a car

Which strikes you as better?

What do we want our sample to look like?

In particular, what relationship do we want between the sample and the population? What steps can we take to improve our odds of success in achieving this?

# Discussion

![talk amongst yourselves](https://media.giphy.com/media/l2SpQRuCQzY1RXHqM/giphy.gif)

The first way of sampling is considered a convenience sample.
You are going about collection in a non-random manner

# Sample Conditions

1. The sampled observations must be independent
    - The sampling method must be random  


2. Sample size distribution:
    - The more skewed the sample the larger samples we need. 
    - n > 30 is considered a large enough sample unless there is extreme skew




## Population v Sample Terminology
Characteristics of populations are called **parameters**

Characteristics of a sample are called **statistics**

A sample statistic is a **point estimate** of the population parameter

![imgsample](./img/sample_stats.png)

# A Simulation to Reinforce Our Definitions

Let's create a population of systolic blood pressure of adult males in Chicago, assuming a mean of 114 mmHg with a standard deviation of 11 mmHg.  We will also assume the adult male population to be 1.5 million. 

It is impossible to measure the systolic blood pressure of every man in Chicago, but let's assume multiple investigations have led to the conclusion numbers avoe. These are therefore estimators of the population parameter.

$\Large\hat\mu = 114$  
$\Large\hat\sigma = 11$




```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pop = int(1.5*10**6)
# Use numpy to generate a normal distribution with the paramters above

fig, ax = plt.subplots()

sns.kdeplot(sys_pop, ax=ax, shade=True)
ax.set_title('Distribution of Adult Male Systolic Blood Pressure')
ax.set_xlabel('Systolic BP')
```




    Text(0.5, 0, 'Systolic BP')




![png](index_files/index_15_1.png)



```python
#__SOLUTION__
# Use numpy to generate a normal distribution of the 
sys_pop = np.random.normal(loc=114, scale=11, size=pop)

fig, ax = plt.subplots()

sns.kdeplot(sys_pop, ax=ax, shade=True)
ax.set_title('Distribution of Adult Male Systolic Blood Pressure')
ax.set_xlabel('Systolic BP')
```




    Text(0.5, 0, 'Systolic BP')




![png](index_files/index_16_1.png)


Let's then imagine we develop an effective manner of random sampling, and simulate with numpy. Our sample size is 40 people.



```python
sample_size = 40
sample = np.random.choice()

# We can look at the distribution of the values in the sample.
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-255-a6c32752e256> in <module>
          1 sample_size = 40
    ----> 2 sample = np.random.choice()
          3 
          4 # We can look at the distribution of the values in the sample.


    mtrand.pyx in numpy.random.mtrand.RandomState.choice()


    TypeError: choice() takes at least 1 positional argument (0 given)



```python
#__SOLUTION__
sample_size = 40
sample = np.random.choice(sys_pop, sample_size)

# We can look at the distribution of the values in the sample.
```


```python
fig, ax = plt.subplots()
sns.distplot(sample, ax=ax, bins=15)
ax.set_title('Sample Distribution of Systolic BP Measurements')
```




    Text(0.5, 1.0, 'Sample Distribution of Systolic BP Measurements')




![png](index_files/index_20_1.png)


We can then calculate the sample statistics:


```python
print(f'Sample mean: {sample.mean()}')
print(f'Sample standard deviation: {sample.std()}')
print(f'Sample median: {np.median(sample)}')
```

    Sample mean: 116.82954837889629
    Sample standard deviation: 13.235768438184078
    Sample median: 116.88056519697305


If we repeated this process, taking samples of the population repeatedly, we would get an array of sample statistics.


```python
number_of_samples = 1000
sample_size = 40
sample_stats = []

for _ in range(number_of_samples):
    # collect the mean of each of the 1000 samples in sample stats
    pass

```


```python
number_of_samples = 1000
sample_size = 50
sample_stats = []

for _ in range(number_of_samples):
    sample = np.random.choice(sys_pop, sample_size)
    # collect the mean of each of the 1000 samples in sample stats
    sample_stats.append(sample.mean())

```

The collection of sample stats represents our __sampling distribution__


```python
fig, ax = plt.subplots()
ax.hist(sorted(sample_stats), bins=20)
ax.set_title('Sampling Distribution\n of Systolic BP')
ax.set_xlabel("Systolic Blood Pressure")
ax.set_ylabel('Count');
```


![png](index_files/index_27_0.png)


An interesting property of this sampling distribution:
    
As we continue to sample, the mean of the sampling distribution gets closer and closer to the population mean.

### Standard Error of the Mean

The standard error of the mean is the standard deviation of the sampling distribution.
The issue is that a sample is not an exact replica of the population. We need to account for that fact in order to make our estimate of the $\mu$ value possible. Let's break it down:

**Population sigma** <br/>

$\large\sigma _{x} = \frac{\sigma }{\sqrt{n}}$

* $ \sigma _{x}$ = standard error of $\bar{x} $
* $ \sigma $ = standard deviation of population

### What is the standard error of the mean for systolic blood pressure example with known mean and standard deviation, assuming a sample size of 40


```python
#__SOLUTION__
11/np.sqrt(40)
```




    1.7392527130926085



**What if we do not know the population sigma?**<br>
If we do not know the population standard deviation, we can approximate it by using the sample standard deviation.

$\large\sigma _{x} ≈ \frac{s}{\sqrt{n}}$

* s = sample standard deviation


```python
#__SOLUTION__
np.random.normal(114, 11, 40).std()/np.sqrt(40)
```




    1.7713767393865683



**Sample size impact on standard error of mean**<br>

How should sample size influence standard error of the mean?

It will get *smaller* as sample size *increases*

![error](./img/diminishing_error.png)  
Important implication: The Standard Error of the mean remains the same as long as the population standard deviation is known and sample size remains the same.



```python
def standard_error(distribution, largest_sample_size, population_std=None):
    
    '''
    Calculate the standard errors for a range of sample sizes
    to demonstrate how standard error decreases when sample 
    size increases.
    '''
 
    std_errors = {}
    
    for sample_size in range(50,largest_sample_size+1):
        sample = np.random.choice(distribution, size=sample_size, replace=True)
        # Standard error with sample distribution standard deviation 
        # in place of population
        if population_std == None:
            std_err = np.std(sample)/np.sqrt(sample_size)
            std_errors[sample_size] = std_err
        
        else:
            std_err = population_std/np.sqrt(sample_size)
            std_errors[sample_size] = std_err
        
    return std_errors
    
```


```python
std_errors = standard_error(sys_pop, 1000)

fig, ax = plt.subplots()

sns.scatterplot(list(std_errors.keys()), list(std_errors.values()))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a3ebe4f60>




![png](index_files/index_37_1.png)



```python
std_errors = standard_error(sys_pop, 1000, population_std=114)

fig, ax = plt.subplots()

sns.scatterplot(list(std_errors.keys()), list(std_errors.values()))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x124a5fac8>




![png](index_files/index_38_1.png)


# Word Exercise 
Put the variables in the correct place.



```python

var_1 = 'population'
var_2 = 'sample'
var_3 = 'point estimate'
var_4 = 'statistic'
var_5 = 'parameter'
var_6 = 'sampling'


print(f"""We sampled 40 bee hives and calcuted the mean colony population 
          to be 75,690 bees. 75,690 is a {var_1} of the population paramter\n""")

print(f"""We repeatedly sample 40 people at random from Seattle and 
        measure their heart rate,then calculate the mean of each sample. 
        We call the plot of this collection of statistics
        the {var_2} distribution.
        """)

print(f"""There are exactly 58 Javan Rhino's left in the wild. 
        Their mean length has been measured accurately at 5 feet.
        This mean length is considered a population {var_3}. 
        """)

print(f"""If we plot a histogram of individual pistil lengths 
      measured on 50 hibiscus flowers, we would be plotting the distribution 
      of an attribute of our {var_4} of hibiscus flowers. 
        """)

print(f"""Since every restaurant in Chicago is required by law to register
        with the city, we can accurately count the number of active pizza restaurants
         operating right now.  This group represents the {var_5} of actively 
        operating, registered pizza restaurants in Chicago.
    """)

print(f"""The mean number of hourly hits to Jelle's Marble Racing website 
            randomly sampled across a seven day period represents a sample
            {var_6}.
        """)
```

    We sampled 40 bee hives and calcuted the mean colony population 
              to be 75,690 bees. 75,690 is a population of the population paramter
    
    We repeatedly sample 40 people at random from Seattle and 
            measure their heart rate,then calculate the mean of each sample. 
            We call the plot of this collection of statistics
            the sample distribution.
            
    There are exactly 58 Javan Rhino's left in the wild. 
            Their mean length has been measured accurately at 5 feet.
            This mean length is considered a population point estimate. 
            
    If we plot a histogram of individual pistil lengths 
          measured on 50 hibiscus flowers, we would be plotting the distribution 
          of an attribute of our statistic of hibiscus flowers. 
            
    Since every restaurant in Chicago is required by law to register
            with the city, we can accurately count the number of active pizza restaurants
             operating right now.  This group represents the parameter of actively 
            operating, registered pizza restaurants in Chicago.
        
    The mean number of hourly hits to Jelle's Marble Racing website 
                randomly sampled across a seven day period represents a sample
                sampling.
            



```python
#__SOLUTION__
# Word Exercise

var_1 = 'population'
var_2 = 'sample'
var_3 = 'point estimate'
var_4 = 'statistic'
var_5 = 'parameter'
var_6 = 'sampling'


print(f"""We sampled 40 bee hives and calcuted the mean colony population 
          to be 75,690 bees. 75,690 is a {var_3} of the population paramter\n""")

print(f"""We repeatedly sample 40 people at random from Seattle and 
        measure their heart rate,then calculate the mean of each sample. 
        We call the plot of this collection of statistics
        the {var_6} distribution.
        """)

print(f"""There are exactly 58 Javan Rhino's left in the wild. 
        Their mean length has been measured accurately at 5 feet.
        This mean length is considered a population {var_5}. 
        """)

print(f"""If we plot a histogram of individual pistil lengths 
      measured on 50 hibiscus flowers, we would be plotting the distribution 
      of an attribute of our {var_2} of hibiscus flowers. 
        """)

print(f"""Since every restaurant in Chicago is required by law to register
        with the city, we can accurately count the number of active pizza restaurants
        actively operating right now.  This group represents the {var_1} of actively 
        operating, registered pizza restaurants in Chicago.
    """)

print(f"""The mean number of hourly hits to Jelle's Marble Racing website 
            randomly sampled across a seven day period represents a sample
            {var_4}.
        """)
```

    We sampled 40 bee hives and calcuted the mean colony population 
              to be 75,690 bees. 75,690 is a point estimate of the population paramter
    
    We repeatedly sample 40 people at random from Seattle and 
            measure their heart rate,then calculate the mean of each sample. 
            We call the plot of this collection of statistics
            the sampling distribution.
            
    There are exactly 58 Javan Rhino's left in the wild. 
            Their mean length has been measured accurately at 5 feet.
            This mean length is considered a population parameter. 
            
    If we plot a histogram of individual pistil lengths 
          measured on 50 hibiscus flowers, we would be plotting the distribution 
          of an attribute of our sample of hibiscus flowers. 
            
    Since every restaurant in Chicago is required by law to register
            with the city, we can accurately count the number of active pizza restaurants
            actively operating right now.  This group represents the population of actively 
            operating, registered pizza restaurants in Chicago.
        
    The mean number of hourly hits to Jelle's Marble Racing website 
                randomly sampled across a seven day period represents a sample
                statistic.
            



# 2. Use numpy to randomly sample a distribution





## Group Exercise

Below, we have four different sample scenarios.  Each group will code out the following: 

You are given a "population" to sample from based on the type of distribution.

1. Take a random sample of size n, where n > 30, from the population and calculate the mean of that population.

2. Repeat the sample n numbers of times (n = 1000). 

3. Plot the sampling distribution


```python
qpe = ['Matt', 'Sindhu', 'Jeff', 'Josh', 
              'Reuben', 'Oz', 'Paul', 'Angelica',
             'Elena', 'Prabhakar', 'Ali', 'Karim', 
             'Sam', 'Hunter', 'Jon', 'Andrew', 'William']

for n in range(1,4):
    group = np.random.choice(qpe, 4, replace=False)
    print(f'group {n}', group)
    for name in list(group):
        qpe.remove(name)

```

    group 1 ['Matt' 'Sindhu' 'Hunter' 'Jon']
    group 2 ['Oz' 'William' 'Prabhakar' 'Angelica']
    group 3 ['Sam' 'Josh' 'Reuben' 'Jeff']


## Group 1:

A bowler on the PBA rolls a strike 60% of the time. The population strikes of all games ever bowled is stored in in the population variable below.



```python
population = np.random.binomial(12, .6, 10000)
fig, ax = plt.subplots()
ax.bar(range(0,12), np.unique(population, return_counts=True)[1])
ax.set_title('Strikes Per Game')
```




    Text(0.5, 1.0, 'Strikes Per Game')




![png](index_files/index_48_1.png)



```python
#__SOLUTION__

sample_means = []
for n in range(1000):
    sample = np.random.choice(population, 50)
    sample_means.append(sample.mean())
    
fig, ax = plt.subplots()
ax.hist(sample_means, bins = 20)
```




    (array([  2.,   6.,   2.,  17.,  30.,  45.,  90., 102., 114., 139., 122.,
            107., 103.,  62.,  32.,  18.,   7.,   1.,   0.,   1.]),
     array([6.46 , 6.539, 6.618, 6.697, 6.776, 6.855, 6.934, 7.013, 7.092,
            7.171, 7.25 , 7.329, 7.408, 7.487, 7.566, 7.645, 7.724, 7.803,
            7.882, 7.961, 8.04 ]),
     <a list of 20 Patch objects>)




![png](index_files/index_49_1.png)


## Group 2:

Stored in the variable below is the number of pieces of mail that arrive per week at your door for each of the 4500 weeks in your life.  


```python
mail_population = np.random.poisson(3, 4500)
counts = np.unique(mail_population, return_counts=True)

fig, ax = plt.subplots()
ax.bar(np.unique(counts[0]), counts[1])
ax.set_title('Distribution of Pieces of Mail/Week')
ax.set_xlabel("Pieces of Mail")
```




    Text(0.5, 0, 'Pieces of Mail')




![png](index_files/index_51_1.png)



```python
#__SOLUTION__

sample_means = []
for n in range(1000):
    sample = np.random.choice(mail_population, 50)
    sample_means.append(sample.mean())
    
fig, ax = plt.subplots()
ax.hist(sample_means, bins = 30)
ax.set_title('Sample Means of Pieces of Mail\n Arriving at your door')
ax.set_xlabel('Number of pieces of mail')
```




    Text(0.5, 0, 'Number of pieces of mail')




![png](index_files/index_52_1.png)


# Group 3 

The population data for the number of minutes between customers arriving in a Piggly Wiggly is stored in the variable piggly_population.


```python
# on average, 20 customers enter per hour
piggly_population = np.random.exponential(1/(20/60), size=10000)
fig, ax = plt.subplots()
ax.hist(piggly_population, bins = 50, normed=True)
ax.set_title('Sample Means of Time Between Piggle Wiggly Customers')
ax.set_xlabel('Minutes');
```

    /Users/johnmaxbarry/.local/lib/python3.7/site-packages/ipykernel_launcher.py:4: MatplotlibDeprecationWarning: 
    The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
      after removing the cwd from sys.path.



![png](index_files/index_54_1.png)



```python
#__SOLUTION__

sample_means = []
for n in range(1000):
    sample = np.random.choice(piggly_population, 50)
    sample_means.append(sample.mean())
    
fig, ax = plt.subplots()
ax.hist(sample_means, bins = 30);
ax.set_title("""Sample means of number of minutes\n between people entering a Piggly Wiggly""")
ax.set_xlabel("Number of minutes between customers")
```




    Text(0.5, 0, 'Number of minutes between customers')




![png](index_files/index_55_1.png)


# 3. Central Limit Theorem

If we take repeated samples of a population, the sampling distribution of sample means will approximate to a normal distribution, no matter the underlying distribution!

## $E(\bar{x_{n}}) = \mu$

as n --> "large"

[good D3 example](https://seeing-theory.brown.edu/probability-distributions/index.html)

[good video demonstration](https://www.youtube.com/watch?v=jvoxEYmQHNM)


Let's look at an example taken from the ubiquitous Iris dataset. This histogram represents the distributions of sepal length:


![probgif](./img/probability-basics.gif)

https://www.kaggle.com/tentotheminus9/central-limit-theorem-animation

As we will see in hypothesis testing, pairing this theorem with the Empirical rule will be very powerful.

![empirical](img/empirical_rule.png)



Knowing that any sampling distribtion, no matter the underlying population distribution, will approach normality, we will be able to judge, given the empirical rule, how rare a given sample statistic is.  

## Bike Example
Capital bike share is trying to figure out their pricing for members versus non-members. The first step in their analysis is to see if members vs non-members ride for different amounts of time per ride.

Let's head over [here](https://s3.amazonaws.com/capitalbikeshare-data/index.html) for some DC bike data!


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
%matplotlib inline

```


```python
! curl https://divvy-tripdata.s3.amazonaws.com/Divvy_Trips_2020_Q1.zip -o 'divy_2020_Q1.zip'
! unzip divy_2020_Q1.zip data
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 15.1M  100 15.1M    0     0  3196k      0  0:00:04  0:00:04 --:--:-- 3213k
    Archive:  divy_2020_Q1.zip
      inflating: data/Divvy_Trips_2020_Q1.csv  
       creating: data/__MACOSX/
      inflating: data/__MACOSX/._Divvy_Trips_2020_Q1.csv  



```python
from src.data_import import prep_divy

divy_trips = prep_divy()

```


```python
# Remember the distribution has heavy right skew. 
# Before proceeding, let's remove outliers which are greater than 3 standard deviations of the mean

```


```python
# Store the number of rides in the original dataframe for comparison
original_number_of_rides = divy_trips.shape[0]
original_number_of_rides
# your code here

```




    426887




```python
#__SOLUTION__
# Remember the distribution has heavy right skew. 
# Before proceeding, let's remove outliers which are greater than 3 standard deviations of the mean

divy_trips = divy_trips[divy_trips.ride_time<divy_trips.ride_time.std()*3]

```


```python
divy_trips.shape[0]
```




    424870




```python
#__SOLUTION__
# How many rides were lost
original_number_of_rides - divy_trips.shape[0]
```




    2017




```python
# Divide the data set into casual and member groups

casual = None
member = None
```


```python
#__SOLUTION__
# Divide the data set into casual and member groups

casual = divy_trips[divy_trips.member_casual == 'casual']
member = divy_trips[divy_trips.member_casual == 'member']
```


```python
# Calculate the mean and standard deviation of each group's ride times

casual_mean_ride_time = casual.ride_time.mean()
member_mean_ride_time = member.ride_time.mean()

casual_std_ride_time = casual.ride_time.std()
member_std_ride_time = member.ride_time.std()

print(f"Casual mean ride time: {casual_mean_ride_time}")
print(f"Member mean ride time: {member_mean_ride_time}")

print(f"Casual std ride time: {casual_std_ride_time}")
print(f"Member mean ride time: {member_std_ride_time}")
```

    Casual mean ride time: 1630.851988107715
    Member mean ride time: 666.4950742759517
    Casual std ride time: 1513.1592710299005
    Member mean ride time: 528.4009595082442



```python
# Using np.random.choice take a sample of 40 ride times from the casual group
np.random.seed(42)
casual_sample = np.random.choice(casual.ride_time, 40)
```


```python
# Calculate the mean and std of the casual sample (sample statistics)
np.random.seed(42)

casual_sample_mean = casual_sample.mean()
casual_sample_std = casual_sample.std()
print(f"casual_sample mean: {casual_sample_mean}")
print(f"casual_sample std: {casual_sample_std}")

```

    casual_sample mean: 2027.35
    casual_sample std: 1862.9139479589496



```python
# Using np.random.choice take a sample of 40 ride times from the member group
np.random.seed(42)

member_sample = np.random.choice(member.ride_time, 40)

```


```python
# Calculate the mean and std of the member sample (sample statistics)
member_sample_mean = member_sample.mean()
member_sample_std = member_sample.std()
print(f"member_sample mean: {member_sample_mean}")
print(f"member_sample std: {member_sample_std}")
```

    member_sample mean: 728.05
    member_sample std: 615.9052260697257



```python
# Repeat the sample 1000 times for both groups, and store the means of each of 1000 samples in two lists.

member_means = []

for _ in range(10000):
    new_sample = np.random.choice(member.ride_time, 40)
    member_means.append(new_sample.mean())

    
```


```python
casual_means = []

for _ in range(10000):
    new_sample = np.random.choice(casual.ride_time, 40)
    casual_means.append(new_sample.mean())

```


```python
# plot the sampling distribution of both the casual and member groups

fig, ax = plt.subplots()

ax.hist(member_means, bins=100)
ax.hist(casual_means, bins=100);
```


![png](index_files/index_80_0.png)


# There are three ways we could calculate the Standard Error of the Mean
    1. Since we are sampling from a large population of rides, we could calculate it using the population std.
    2. Since we used numpy to randomly sampled the means, we could take the mean of the sample (should be very close to the number calculated above)
    3. We could use our original samples to approximate the SEM. 



```python
# 1. Calculate it using the population std for both member and casual.

```


```python
#__SOLUTION__
# 1. Calculate it using the population std for both member and casual.

sem_member = member.ride_time.std()/np.sqrt(40)
sem_casual = casual.ride_time.std()/np.sqrt(40)

print(f"SEM Member: {sem_member}")
print(f"SEM Casual: {sem_casual}")
```

    SEM Member: 83.54752749322284
    SEM Casual: 239.2514879527262



```python
# 2. Calculate the standard error of the mean of both populations using the randomly generated samples

```


```python
#__SOLUTION__
# 2. Calculate the standard error of the mean of both populations using the randomly generated samples

sem_member = np.std(member_means)
sem_casual = np.std(casual_means)
print(f"Standard error of mean (member): {sem_member}")
print(f"Standard error of mean (casual): {sem_casual}")
```

    Standard error of mean (member): 82.83579266070582
    Standard error of mean (casual): 241.2689116902536



```python
# 3. Use our samples to approximate the SEM for both casual and member
```


```python
#__SOLUTION__
sem_approximate_casual = casual_sample_mean/np.sqrt(40)
```


```python
#__SOLUTION__
sem_approximate_member = member_sample_mean/np.sqrt(40)
```


```python
# Calculate the z_score of the casual sample wrt the member sampling distribution
```


```python
# Your Code Here
```




    666.5359925




```python
#__SOLUTION__

(casual_sample.mean() - np.mean(member_means))/sem_member

```




    16.427850374703045


