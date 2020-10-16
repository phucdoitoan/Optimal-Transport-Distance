# Optimal Transport Distance

Pytorch Solvers for the:\
+) Entropic Regularized Wasserstein distance \[1\]\[2\] \
+) Entropic Regularized Gromov-Wasserstein distance \[1\] \[3\] 

Examples:
Points clouds x, y are the rotation of each other.
<p align="center">
  <img src="https://github.com/phucdoitoan/Optimal_Transport_Distance/blob/main/simple_rotation_points.png" width="450" title="the two points clouds">
</p>

Transport plan learned by Gromov-Wasserstein distance. \
+) Relatively large value of eps => denser plan P
<p align="center">
  <img src="https://github.com/phucdoitoan/Optimal_Transport_Distance/blob/main/simple_rotation_P_unstable.png" width="450" title="Relatively large of eps">
</p>

+) Relatively small value of eps => sparse plan P
<p align="center">
  <img src="https://github.com/phucdoitoan/Optimal_Transport_Distance/blob/main/simple_rotation_P_stable.png" width="450" title="Relatively small of eps">
</p>

Reference:

\[1\] [Computational Optimal Transport](https://arxiv.org/abs/1803.00567) \
\[2\] [Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895) \
\[3\] [Gromov-wasserstein averaging of kernel and distance matrices](http://proceedings.mlr.press/v48/peyre16.html) \

The codes are also refered from: \
\[4\] https://github.com/gpeyre/SinkhornAutoDiff \
\[5\] https://github.com/PythonOT/POT/blob/master/ot/gromov.py
