# ML-DSBA-AI-Assignment_2

Assignment propositon for Foundation of Machine Learning courses at CentraleSupélec.  

Proposed by Robin Schwob and Paul Asquin.  
This assigment is based on the Kaggle contest : [www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)  

# Run the models
It's possible to run each model independently bu using the test implemented in test.py. Run the following instrucions to execute specific models.  
You can also test the load data function by running:  
```
python3 test.py -m test_load_data
```

## Basic classifier
*Test the basic classifier given*
```
python3 test.py -m test_basic_classifier
```

## Decision tree
*Testing Decision Tree for different depths (best result with D=5)*
```
python3 test.py -m test_decision_tree
```


## Ada boost
*Adaboost Test for Different values of D (best with D=2)*
```
python3 test.py -m test_ada_boot
```

## NN
*NN Test with sgd, different constant lr, 1 hidden layer of varying size*
```
python3 test.py -m test_NN_1
```
  
*NN test for higher hidden layer sizes (from 200 to 400)*
```
python3 test.py -m test_NN_2
```