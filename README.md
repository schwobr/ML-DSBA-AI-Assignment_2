# ML-DSBA-AI-Assignment_2

Assignment proposition for Foundation of Machine Learning courses at CentraleSupélec.  

Proposed by Robin Schwob and Paul Asquin.  
This assignment is based on the Kaggle contest : [www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)  

# Run the models
It's possible to run each model independently bu using the test implemented in test.py. Run the following instructions to execute specific models.  
You can also test the load_data function by running:  
```
python3 test.py -m test_load_data
```

And you can test the load_data_panda function with:
```
python3 test.py -m test_load_data_panda
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

## LDA
```
python3 test.py -m test_LDA
```

## SVM
```
python3 test.py -m test_SVM
```

## KNN
```
python3 test.py -m test_KNN
```

## Random Forest
```
python3 test.py -m test_random_forest
```