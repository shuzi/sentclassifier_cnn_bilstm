require 'nn'


torch.setnumthreads(1)
torch.manualSeed(1)
math.randomseed(1)



model = nn.Sequential()
model:add(nn.Linear(10,1))
model:zeroGradParameters()
criterion=nn.AbsCriterion()

print(model:get(1).gradWeight)

target=torch.randn(1)
target[1]=2
input=torch.ones(1,10)
output=model:forward(input)
criterion:forward(output, target)
df_do = criterion:backward(output, target)
model:backward(input,df_do)
print(model:get(1).gradWeight)

model:zeroGradParameters()

target=torch.randn(2)
target[1]=2
target[2]=2
input=torch.ones(2,10)
output=model:forward(input)
criterion:forward(output, target)
df_do = criterion:backward(output, target)
model:backward(input,df_do)
print(model:get(1).gradWeight)



