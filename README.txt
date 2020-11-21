The implementation of MTAdam can be found in the file mtadam.py. An example of using the MTAdam optimizer can be found in main.py. The main.py file uses MTAdam and compares it to Adam with optimal and suboptimal weights.

The main.py file executes a sequence of MNIST experiments. As can be seen in the file, the use of MTAdam is as simple as using Adam, and requires the following steps: (a) initiating the MTAdam optimizer (in a similar way to Adam). (b) keeping the multi-term loss objective decomposed as a sequence of single terms, and sending the sequence as an argument to MTAdam.step(). (c) avoid calling the function loss.backward(), since it is done internally in MTAdam.step().

Dependencies:
Pytorch version >= 1.0
tqdm
torchvision

Files:
mtadam.py
Main.py

If you find the code useful in your research, please consider citing the paper:
@article{malkiel2020mtadam,
  title={MTAdam: Automatic Balancing of Multiple Training Loss Terms},
  author={Malkiel, Itzik and Wolf, Lior},
  journal={arXiv preprint arXiv:2006.14683},
  year={2020}
}

