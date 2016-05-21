#!/opt/share/torch-7.0/bin/th
require 'torch'
--require 'socket'
arg={}
io.write("HI WEI!!!")
function prepResult(str_len)
   local res = {}
   cnt = 0;
   for i = 1, 300 do
      if cnt >= 300 then
	break
      end
      res[i]=str_len
      if i == 1 then
	 --print("[Lua]: str_len", res[#res])
      end
      cnt = cnt + 1
   end
   return res
end

cmd = torch.CmdLine('_')
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 1, 'number of threads')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-batchSizeTest', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:option('-trainFile', 'train', 'input training file')
cmd:option('-validFile', 'valid', 'input validation file')
cmd:option('-testFile', 'test', 'input test file')
cmd:option('-embeddingFile', 'embedding', 'input embedding file')
cmd:option('-embeddingDim', 100, 'input word embedding dimension')
cmd:option('-contConvWidth', 2, 'continuous convolution filter width')
cmd:option('-wordHiddenDim', 200, 'first hidden layer output dimension')
cmd:option('-numFilters', 1000, 'CNN filters amount')
cmd:option('-hiddenDim', 1000, 'second hidden layer output dimension')
cmd:option('-numLabels', 311, 'label quantity')
cmd:option('-epoch', 200, 'maximum epoch')
cmd:option('-L1reg', 0, 'L1 regularization coefficient')
cmd:option('-L2reg', 1e-4, 'L2 regularization coefficient')
cmd:option('-trainMaxLength', 100, 'maximum length for training')
cmd:option('-testMaxLength', 100, 'maximum length for valid/test')
cmd:option('-trainMinLength', 40, 'maximum length for training')
cmd:option('-testMinLength', 40, 'maximum length for valid/test')
cmd:option('-gradClip', 0.5, 'gradient clamp')
cmd:option('-gpuID', 1, 'GPU ID')
cmd:option('-outputprefix', 'none', 'output file prefix')
cmd:option('-prevtime', 0, 'time start point')
--cmd:option('-loadmodel', 'none', 'load model file name')
cmd:option('-loadmodel', 'model_0000026.01_model', 'load model file name')
cmd:text()
opt = cmd:parse(arg or {})
print(opt)
--opt.rundir = cmd:string('experiment', opt, {dir=true})
--paths.mkdir(opt.rundir)
--cmd:log(opt.rundir .. '/log', params)

-- Beginning of getting weiarg 
-- print "Wei args beginning"
-- for i =1, table.getn(weiarg) do
--    print (i, weiarg[i])
-- end
-- print "Wei args ending"
-- End of getting weiarg

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
validDataTensor = torch.Tensor()
validDataTensor_y = {}
testDataTensor = torch.Tensor()
testDataTensor_y = {}


dofile 'prepareData.lua'
if opt.type == 'cuda' then
  trainDataTensor =  trainDataTensor:cuda()
  trainDataTensor_y =  trainDataTensor_y:cuda()
  validDataTensor = validDataTensor:cuda()
  testDataTensor = testDataTensor:cuda()
end
dofile 'train.lua'
collectgarbage()
collectgarbage()

sys.tic()
validState = {}
testState = {}
--t1 = socket.gettime()*1000;
loadModel(opt.loadmodel)
--print ("loading time (milli-sec)", socket.gettime()*1000 - t1);
--test(validDataTensor, validDataTensor_y, validState)
--test(testDataTensor, testDataTensor_y, testState)
--print("About to yield ... \n");
--arg={}
counter=0

function mycoroutine() -- wrap a coroutine so that it can be yielded more convinietnly
while (1)
do
   counter = counter + 1
   --local sum = -5
   local str_len = -1
   for i=1,table.getn(arg) do
      --print(i,arg[i])
      str_len = string.len(arg[i])
      --print("[Lua] args from c:", i, arg[i])
   end
   --print("hi from Wei")
   --return 100,234,456,678
   --print("about to yield ");
   --test(testDataTensor, testDataTensor_y, testState)
   coroutine.yield(unpack(prepResult(str_len)));
end
end

