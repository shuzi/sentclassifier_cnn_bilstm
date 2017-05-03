
require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'
cutorch.manualSeedAll(2)

inputDim=8
outputDim=13
attention_da=31
attention_r=37
batchsize=2

model = nn.Sequential()
model:add(cudnn.BLSTM(inputDim, outputDim, 1, true))


attention = nn.Sequential()
attention:add(nn.Copy(nil,nil,true))
attention:add(cudnn.TemporalConvolution(outputDim*2, attention_da, 1))
attention:add(cudnn.Tanh())
attention:add(cudnn.TemporalConvolution(attention_da, attention_r, 1))
attention:add(nn.Transpose({2,3}))
attention:add(nn.View(batchsize*attention_r, -1))
attention:add(nn.SoftMax())
attention:add(nn.View(batchsize, attention_r, -1))
part1 = nn.ConcatTable()
part1:add(attention)
part1:add(nn.Identity())
model:add(part1)
model:add(nn.MM())
model:add(nn.View(batchsize, outputDim*2*attention_r))
model:add(nn.ReLU())
model:add(nn.Linear(outputDim*2*attention_r, 1000))




model = model:cuda()

input = torch.randn(2,10,inputDim):contiguous():cuda()

p,g=model:getParameters()
print(model:get(1).weight[1])
print(p[1])
model:get(1).weight[1]=22
print(model:get(1).weight[1])
print(p[1])



--print(model:forward(input))



