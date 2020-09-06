# rinn
Some of the code and all of the simulated data used in my Redundant Input Neural Network (RINN) work. 

See rinn_example.py and constrained_GA_example.py for simple examples of RINN and the constrained GA (i.e., ES-C). 
RINN was written with older versions of TensorFlow and works here (with warnings) with TensorFlow 1.14. It has not been updated
for later versions of TensorFlow, but probably works with some earlier versions :). 

To run from the command-line (maybe using conda environment with requirements installed):
```
python rinn_example.py
```

```
python constrained_GA_example.py
```

See our RINN paper presented at the 2020 KDD Workshop on Causal Discovery:  http://proceedings.mlr.press/v127/young20a.html

For running on GPU server:
```
CUDA_VISIBLE_DEVICES=0 nohup python -u model_selection_example.py &> model_selection.out
```
