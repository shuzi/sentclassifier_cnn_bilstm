require 'nn'


torch.setnumthreads(1)
torch.manualSeed(1)
math.randomseed(1)



model = nn.Sequential()


input=torch.randn(2,7,8)

print(input)
print(nn.Linear(8,2):forward(nn.Mean(2):forward(input)))


