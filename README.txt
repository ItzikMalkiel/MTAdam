The implementation of MTAdam can be found in the file mtadam.py. An example of using MTAdam optimizer can be found in main.py. Specifically, the main.py file uses MTAdam, and compares it to unbalanced and balanced Adam.

The main.py file incorporates a sequence of MNIST experiments. As can be seen in the file, the use of MTAdam is as simple as using Adam, and requires the following steps: (a) initiating the MTAdam optimizer (in a similar way to Adam). (b) keeping the multi-term loss objective decomposed as a sequence of single terms, and sending the sequence as an argument to MTAdam.step(). (c) avoiding calling to loss.backward(), since it is done internally in MTAdam.step().

Dependencies:
Pytorch version >= 1.0
tqdm
torchvision

Files:
mtadam.py
Main.py



