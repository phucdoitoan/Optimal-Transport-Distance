


import torch
import torch.nn as nn
from wasserstein import Entropic_Wasserstein, Stabilized_Entropic_Wasserstein


def intra_cost_matrix(x, cost_type='L2'):

	if cost_type == 'L2':
		x_row = x.unsqueeze(-2)
		x_col = x.unsqueeze(-3)

		C = torch.sum((x_row - x_col) ** 2, dim=-1)
	else:
		raise NotImplementedError('The cost type %s is not implemented!' %(cost_type))

	return C


class Entropic_GromovWasserstein(nn.Module):

	"""
	Computed the entropic regularized gromov-wassertsein discrepancy

	Reference:
		Computational Optimal Transport, chapter 10.6.3, 10.6.4
		Gromov-Wasserstein Averaging of Kernel and Distance Matrices, Peyre et al ICML 2016
	"""

	def __init__(self, eps, max_iter, thresh, w_max_iter, w_thresh, inter_loss_type='square_loss', stable_sinkhorn=False, verbose=False):

		super(Entropic_GromovWasserstein, self).__init__()

		self.eps = eps
		self.max_iter = max_iter
		self.thresh = thresh

		self.inter_loss_type = inter_loss_type

		self.verbose = verbose

		if stable_sinkhorn:
			self.Entropic_W = Stabilized_Entropic_Wasserstein(eps, w_max_iter, w_thresh, verbose=verbose)
		else:
			self.Entropic_W = Entropic_Wasserstein(eps, w_max_iter, w_thresh, verbose=verbose)

	def forward(self, x, y, px, py, intra_loss_type='L2', dtype='double'):
		if dtype == 'double':
			x = x.double()
			y = y.double()
			px = px.double()
			py = py.double()            
		else:
			pass

		Cx = intra_cost_matrix(x, cost_type=intra_loss_type)
		Cy = intra_cost_matrix(y, cost_type=intra_loss_type)

		return self.forward_with_cost_matrices(Cx, Cy, px, py)

	def forward_with_cost_matrices(self, Cx, Cy, px, py):
		
		nx, ny = Cx.shape[0], Cy.shape[0]
		P = px.unsqueeze(-1) * py.unsqueeze(-2)

		f1, f2, h1, h2 = self.func_define(inter_loss_type=self.inter_loss_type)

		Cxy = (f1(Cx) @ px.reshape(-1, 1)).repeat((1, ny)) + (py.reshape(1, -1) @ f2(Cy).T).repeat((nx, 1))

		for it in range(self.max_iter):
			P_old = P

			L = Cxy - h1(Cx) @ P @ h2(Cy).T
			L = 2*L  # Proposition 2 (eq (9)) of Perey et al miss a 2 factor (ref. https://github.com/PythonOT/POT/blob/master/ot/gromov.py)

			_, P = self.Entropic_W.forward_with_cost_matrix(L, px, py)

			err = torch.norm(P - P_old)
			if err < self.thresh:
				if self.verbose:
					print('Break in Gromov-Wasserstein at %s-th iteration: Err = %f' %(it, err))

				break

			if self.verbose:
				if it % 10 == 0:
					print('Iter: %s | Err = %f' %(it, err))

		gw_cost = torch.sum(P * L)

		return gw_cost, P


	def func_define(self, inter_loss_type):
		"""
		Define functions f1, f2, h1, h2 to compute the tensor-matrix multiplication as in Proposition 1 of Peyre et al
		"""

		if inter_loss_type == 'square_loss':
			
			def f1(a):
				return a**2
			def f2(b):
				return b**2
			def h1(a):
				return a
			def h2(b):
				return 2*b 

			return f1, f2, h1, h2

		elif inter_loss_type == 'kl_loss':
			pass
		else:
			raise NotImplementedError('Inter loss type %s is not implemented!' %(inter_loss_type))

