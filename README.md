This is an implementation of decision tree with support for rule post pruning. The necessary files are the following 
    main.py            (main file with options)
    dtree.py           (dependency for representing decision tree)
    learner.py         (dependency for building the decision tree and pruning)
    utils.py           (dependency for utility functions used)
    testTennis.py      (main tennis experiment file)
    testIris.py        (main Iris experiment file)
    testIrisNoisy.py   (main Iris with Noise experiment file)
    data/              (directory of all required attribute, training and testing data files)

No need to compile since the Python files are interpreted

To run the tree with options, use the following command:

$ python3 main.py -a attribute_file.txt -d training_file.txt -t testing_file.txt --prune --debug

where python3 is the python 3.X.X interpreter, 
    attribute_file is the path to the attribute file for the tree (required)
    training_file is the path to the training file for the tree (required)
    testing_file is the path to the testing file for the tree (required)
    debug is the toggle for debug mode and allows for data logging (optional)
    prune is the toggle for pruning the tree after building (optional)

To find out about the options, use:
$ python3 main.py -h 

To run the different experiment files, use the following  command:

$ python3 testTennis.py 
$ python3 testIris.py
$ python3 testIrisNoisy.py

where python3 is the python 3.X.X interpreter, and provided the data files are present 
and in the same directory as the experiment files



