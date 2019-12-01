import torch
import torch.nn.functional as F
w = torch.rand(256)
b = torch.rand(512)
x = torch.rand((2, 256, 7, 7))
#               b    c   h  w


# F.instance_norm(x, weight=w, bias=b)


for i in range(x.size(0)):
    F.instance_norm(x[i].unsqueeze(0), weight=w[i], bias=b[i])
    print("a")

i1 = torch.nn.InstanceNorm2d(256, affine=True)
print(i1.__dict__)
i1._parameters["weight"] = w
i1._parameters["bias"] = b
# print(i1._parameters["weight"].size())

i1(x)
