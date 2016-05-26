#!/opt/share/torch-7.0/bin/th
require 'rnn'
dofile('AddConstantNeg.lua')
dofile('options.lua')

local stringx = require 'pl.stringx'
local optmap = {}
for line in io.lines("configure") do
   local option = stringx.split(line, '=')
   if option[1] == "modelSize" then
      print("Delete previous modelSize!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
   elseif option[1] == "totalUpdateTimes" then
      print("Delete previous totalUpdateTimes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
   else
      optmap[option[1]] = option[2]
   end
end
fp = io.open("configure", "w+" )
for k,v in pairs(optmap) do
   fp:write(string.format("%s=%s\n", k,v ))
end
fp:close()

if io.open("configure", "r") then
   for line in io.lines("configure") do
      local option = stringx.split(line, '=')
      if type(opt[option[1]]) == 'number' then
         opt[option[1]] = tonumber(option[2])
      elseif type(opt[option[1]]) == 'boolean' then
         if option[2] == "true" then
            opt[option[1]] = true
         elseif option[2] == "false" then
            opt[option[1]] = false
         end
      elseif type(opt[option[1]]) == 'string' then
         opt[option[1]] = option[2]
      end
   end
end

print(opt)
--opt.rundir = cmd:string('experiment', opt, {dir=true})
--paths.mkdir(opt.rundir)
--cmd:log(opt.rundir .. '/log', params)

if opt.type == 'cuda' then
  fbok,_ = pcall(require, 'fbcunn')
  if fbok then
     require 'fbcunn'
  end

  cudnnok,_ = pcall(require, 'cudnn')
  if cudnnok then
     require 'cudnn'
     cudnn.fastest = true
     cudnn.benchmark = true
     cudnn.verbose = true
  end
else
   fbok = false
   cudnnok = false
end


if opt.usefbcunn == false then
  fbok = false
elseif opt.usefbcunn == true and fbok == true then
  fbok = true
else
  print("Error: fbcunn is not available to use.")
  fbok = false
end


if opt.usecudnn == false then
  cudnnok = false
elseif opt.usecudnn == true and cudnnok == true then
  cudnnok = true
else
  print("Error: cudnn is not available to use.")
  cudnnok = false
end



if opt.type == 'float' then
   print('==> switching to floats')
   require 'torch'
   require 'nn'
   require 'optim'
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cutorch'
   require 'cunn'
   require 'optim'
   torch.setdefaulttensortype('torch.FloatTensor')
   cutorch.setDevice(opt.gpuID)
   print('GPU DEVICE ID = ' .. cutorch.getDevice())
end
print("####")
print("default tensor type"  .. torch.getdefaulttensortype())
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
math.randomseed(opt.seed)
mapWordIdx2Vector = torch.Tensor()
mapWordStr2WordIdx = {}
mapWordIdx2WordStr = {}
trainDataSet = {}
validDataSet = {}
testDataSet = {}
trainDataTensor = torch.Tensor()
trainDataTensor_y = torch.Tensor()
trainDataTensor_lstm_fwd = torch.Tensor()
trainDataTensor_lstm_bwd = torch.Tensor()
validDataTensor = torch.Tensor()
validDataTensor_lstm_fwd = torch.Tensor()
validDataTensor_lstm_bwd = torch.Tensor()
validDataTensor_y = {}
testDataTensor = torch.Tensor()
testDataTensor_lstm_fwd = torch.Tensor()
testDataTensor_lstm_bwd = torch.Tensor()
testDataTensor_y = {}

dofile 'prepareData.lua'
if opt.type == 'cuda' then
  trainDataTensor =  trainDataTensor:cuda()
  trainDataTensor_y =  trainDataTensor_y:cuda()
  trainDataTensor_lstm_fwd = trainDataTensor_lstm_fwd:cuda()
  trainDataTensor_lstm_bwd = trainDataTensor_lstm_bwd:cuda()
  validDataTensor_lstm_fwd = validDataTensor_lstm_fwd:cuda()
  validDataTensor_lstm_bwd = validDataTensor_lstm_bwd:cuda()
  testDataTensor_lstm_fwd = testDataTensor_lstm_fwd:cuda()
  testDataTensor_lstm_bwd = testDataTensor_lstm_bwd:cuda()
  validDataTensor = validDataTensor:cuda()
  testDataTensor = testDataTensor:cuda()
end

if opt.model == 1 then
   dofile 'model_parallel_cnn_bilstm.lua'
elseif opt.model == 2 then
   dofile 'model_bilstm.lua'
elseif opt.model == 3 then
   dofile 'model_stack_cnn_bilstm.lua'
elseif opt.model == 4 then
   dofile 'model_stack_bilstm_cnn.lua'
elseif opt.model == 5 then
   dofile 'model_cnn.lua'
elseif opt.model == 6 then
   dofile 'model_birnn.lua'
elseif opt.model == 7 then
   dofile 'model_bigru.lua'
end
collectgarbage()
collectgarbage()
modelSize = parameters:size()[1]
totalUpdateTimes = (trainDataTensor:size()[1]/opt.batchSize) * opt.epoch
fp = io.open("configure", "a+" )
fp:write(string.format("modelSize=%s\n", modelSize))
fp:write(string.format("totalUpdateTimes=%s\n", totalUpdateTimes))
fp:close()


