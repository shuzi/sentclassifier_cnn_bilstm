require 'nn'
require 'rnn'

torch.setnumthreads(1)
torch.manualSeed(1)
math.randomseed(1)

model = nn.Sequential()

L_cnn = nn.LookupTableMaskZero(50, 7)
model = nn.Sequential()
model:add(L_cnn)
model:add(nn.View(2, -1, 1, 7))
model:add(nn.Transpose({2,4}))

input=torch.Tensor{{1,2,3,4,5}, {5,4,3,2,1}}


t_layer =  nn.Sequential()
t_layer:add(nn.AddConstant(-2))
t_layer:add(nn.Sigmoid())
t_layer:add(nn.Padding(4,2))

c_layer =  nn.Sequential()
c_layer:add(nn.AddConstant(-2))
c_layer:add(nn.Sigmoid())
c_layer:add(nn.Padding(4,2))
c_layer:add(nn.MulConstant(-1))
c_layer:add(nn.AddConstant(1))

p_layer = nn.Sequential()
p_layer:add(nn.ReLU())
p_layer:add(nn.Padding(4,2))



conv = nn.SpatialConvolution(7, 7, 3, 1, 1, 1, 0, 0)
conv.weight:uniform(-0.01, 0.01)
conv.bias:zero()
model:add(nn.ConcatTable():add(conv):add(nn.Identity()))
model:add(nn.ParallelTable():add(nn.ConcatTable():add(p_layer):add(t_layer):add(c_layer) ):add(nn.Identity()))
model:add(nn.FlattenTable())
model:add(nn.ConcatTable():add(nn.NarrowTable(1,2)):add(nn.NarrowTable(3,2)))
model:add(nn.ParallelTable():add(nn.CMulTable()):add(nn.CMulTable()))
model:add(nn.CAddTable())

print(model)
r=model:forward(input)

print(r)

