#!/opt/share/torch-7.0/bin/th
require 'rnn'
dofile('AddConstantNeg.lua')
--dofile('ArgMax.lua')
dofile('options.lua')
dofile('optim-msgd.lua')
local stringx = require 'pl.stringx'
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
elseif opt.model == 8 then
   dofile 'model_cnn_cnn.lua'
elseif opt.model == 9 then
   dofile 'model_cnn_cnn_cnn.lua'
end
collectgarbage()
collectgarbage()


sys.tic()
epoch = 1
validState = {}
testState = {}
while epoch <= opt.epoch do
   train()
   test(validDataTensor, validDataTensor_lstm_fwd, validDataTensor_lstm_bwd, validDataTensor_y, validState)
   test(testDataTensor, testDataTensor_lstm_fwd, testDataTensor_lstm_bwd, testDataTensor_y, testState)
   if opt.outputprefix ~= 'none' then
      if opt.saveMode == 'last' and epoch == opt.epoch then
         local t = sys.toc()
         saveModel(t + opt.prevtime)
         local obj = {
            em = model:get(1).weight,
            s2i = mapWordStr2WordIdx,
            i2s = mapWordIdx2WordStr
         }
         torch.save(opt.outputprefix .. string.format("_%010.2f_embedding", t + opt.prevtime), obj)
      elseif opt.saveMode == 'every'  then
         local t = sys.toc()
         saveModel(t + opt.prevtime)
         local obj = {
            em = model:get(1).weight,
            s2i = mapWordStr2WordIdx,
            i2s = mapWordIdx2WordStr
         }
         torch.save(opt.outputprefix .. string.format("_%010.2f_embedding", t + opt.prevtime), obj)
      end
   end
   epoch = epoch + 1
end

