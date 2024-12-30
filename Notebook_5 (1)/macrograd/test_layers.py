"""You can use these tests to check your implementation.
It compares your gradients with the one computed by pytorch. If your
implementation is correct, they should match up."""
import unittest
from layers import Linear, ReLU, Sigmoid, BinaryCrossEntropy, CrossEntropy
from torch.nn import Linear as LinearTorch
from torch.nn import ReLU as ReLUTorch
from torch.nn import Sigmoid as SigmoidTorch
from torch.nn import BCELoss as BCELossTorch
from torch.nn import CrossEntropyLoss as CrossEntropyLossTorch
import torch
from utils import seeding

seeding(42)


class TestCompareToPyTorch(unittest.TestCase):
    def test_linear(self):
        i, o = 5, 10

        weight = torch.randn((o, i))
        bias = torch.ones(o)
        with torch.no_grad():
            l_own = Linear(i, o)
            # this depends on how you use your weight, so removing
            # .T might make it work
            l_own.weight.copy_(weight.T)
            l_own.bias.copy_(bias)
            l_pytorch = LinearTorch(i, o)
            l_pytorch.weight.copy_(weight)
            l_pytorch.bias.copy_(bias)

        inp = torch.rand(100, i)
        out_pytorch = l_pytorch(inp)
        out_own = l_own(inp)

        self.assertTrue(
            torch.allclose(out_own, out_pytorch, rtol=1e-5, atol=1e-5)
        )

        out_own.backward(torch.ones_like(out_own))
        out_pytorch.backward(torch.ones_like(out_pytorch))

        self.assertTrue(
            torch.allclose(l_own.weight.grad.T, l_pytorch.weight.grad)
        )
        self.assertTrue(torch.allclose(l_own.bias.grad, l_pytorch.bias.grad))

    def _test_function(self, l_own, l_pytorch):

        inp_own = torch.randn(100, 10)
        inp_pytorch = torch.clone(inp_own)

        inp_own.requires_grad_()
        inp_pytorch.requires_grad_()

        out_own = l_own(inp_own)
        out_pytorch = l_pytorch(inp_pytorch)

        self.assertTrue(torch.allclose(out_own, out_pytorch))

        out_own.backward(torch.ones_like(out_own))
        out_pytorch.backward(torch.ones_like(out_pytorch))

        self.assertTrue(torch.allclose(inp_own.grad, inp_pytorch.grad))

    def _test_loss_function(self, l_own, l_pytorch):
        gt_own = torch.nn.functional.one_hot(
            torch.randint(0, 10, (100,)), num_classes=10
        ).float()
        gt_pytorch = torch.clone(gt_own)

        inp_own = torch.rand(100, 10)
        inp_pytorch = torch.clone(inp_own)
        inp_pytorch2 = torch.clone(inp_own)
        inp_pytorch3 = torch.clone(inp_own)

        inp_own.requires_grad_()
        inp_pytorch.requires_grad_()
        inp_pytorch2.requires_grad_()
        inp_pytorch3.requires_grad_()

        out_own = l_own(inp_own, gt_own)
        out_pytorch = l_pytorch(inp_pytorch, gt_pytorch)
        out_pytorch2 = -torch.sum(
            gt_pytorch * torch.log_softmax(inp_pytorch2, dim=1), dim=1
        ).mean()
        out_pytorch3 = -torch.sum(
            gt_pytorch * torch.log(torch.softmax(inp_pytorch3, dim=1)), dim=1
        ).mean()

        self.assertTrue(torch.allclose(out_own, out_pytorch, atol=1e-7))

        out_own.backward(torch.ones_like(out_own))
        out_pytorch.backward(torch.ones_like(out_pytorch))
        out_pytorch2.backward(torch.ones_like(out_pytorch2))
        out_pytorch3.backward(torch.ones_like(out_pytorch3))

        self.assertTrue(torch.allclose(inp_own.grad, inp_pytorch.grad))

    def test_relu(self):
        self._test_function(ReLU.apply, ReLUTorch())

    def test_sigmoid(self):
        self._test_function(Sigmoid.apply, SigmoidTorch())

    def test_bce(self):
        gt_own = torch.randint(0, 1, (100, 1)).float()
        gt_pytorch = torch.clone(gt_own)

        inp_own = torch.rand(100, 1)
        inp_pytorch = torch.clone(inp_own)

        inp_own.requires_grad_()
        inp_pytorch.requires_grad_()

        out_own = BinaryCrossEntropy.apply(inp_own, gt_own)
        out_pytorch = BCELossTorch()(inp_pytorch, gt_pytorch)

        self.assertTrue(torch.allclose(out_own, out_pytorch, atol=1e-7))

        out_own.backward(torch.ones_like(out_own))
        out_pytorch.backward(torch.ones_like(out_pytorch))

        self.assertTrue(torch.allclose(inp_own.grad, inp_pytorch.grad))

    def test_cross_entropy(self):

        gt_own = torch.nn.functional.one_hot(
            torch.randint(0, 10, (100,)), num_classes=10
        ).float()
        gt_pytorch = torch.clone(gt_own)

        inp_own = torch.rand(100, 10)
        inp_pytorch = torch.clone(inp_own)

        inp_own.requires_grad_()
        inp_pytorch.requires_grad_()

        out_own = CrossEntropy.apply(inp_own, gt_own)
        out_pytorch = CrossEntropyLossTorch()(inp_pytorch, gt_pytorch)

        self.assertTrue(torch.allclose(out_own, out_pytorch, atol=1e-7))

        out_own.backward(torch.ones_like(out_own))
        out_pytorch.backward(torch.ones_like(out_pytorch))

        self.assertTrue(torch.allclose(inp_own.grad, inp_pytorch.grad))


if __name__ == "__main__":
    unittest.main()
