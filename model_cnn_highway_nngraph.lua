dofile('optim-rmsprop-single.lua')
require 'nngraph'


lookup = nn.LookupTableMaskZero(mapWordIdx2Vector:size()[1], opt.embeddingDim)()
lookup.data.module.weight:sub(2,-1):copy(mapWordIdx2Vector)

transdim = nn.View(opt.batchSize, -1, opt.highwayDim)(nn.Linear(opt.embeddingDim, opt.highwayDim)(nn.View(-1, opt.embeddingDim)(lookup)))
transdim2 = nn.Transpose({2,4})(nn.View(opt.batchSize, -1, 1, opt.highwayDim)(transdim))

pw = opt.contConvWidth - 1 - 2*opt.padW
local inputone = nn.Identity()(transdim2)
local inputs = {[1]=inputone}


for i = 1, opt.hwlayers do
  if cudnnok then
    conv = cudnn.SpatialConvolution(opt.highwayDim, opt.highwayDim, opt.contConvWidth, 1, 1, 1, opt.padW, 0)
  else
    conv = nn.SpatialConvolution(opt.highwayDim, opt.highwayDim, opt.contConvWidth, 1, 1, 1, opt.padW, 0)
  end
  conv.weight:uniform(-0.01, 0.01)
  conv.bias:zero()
  
  output = nn.Padding(4, pw)(nn.ReLU()(conv(inputs[i])))
  transform_gate = nn.Padding(4, pw)(nn.Sigmoid()(nn.AddConstant(opt.tgbias)(conv(inputs[i]))))
  carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))
  
  out = nn.CAddTable()({
      nn.CMulTable()({transform_gate, output}),
      nn.CMulTable()({carry_gate, inputs[i]})  })
  table.insert(inputs, out)

end

convback = nn.View(opt.batchSize, -1, opt.highwayDim)(nn.Transpose({2,4})(inputs[opt.hwlayers+1]))
topk = nn.View(opt.batchSize, -1)(nn.TopK(opt.topk, 2, true, true)(convback))
node = nn.ReLU()(nn.Linear(opt.highwayDim*opt.topk, opt.hiddenDim)(topk))

if opt.dropout> 0 then
  node = nn.Dropout(opt.dropout)(node)
end

prob = nn.LogSoftMax()(nn.Linear(opt.hiddenDim, opt.numLabels)(node))

model = nn.gModule({lookup},{prob})

criterion = nn.ClassNLLCriterion()


if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end
if model then
   parameters,gradParameters = model:getParameters()
   print("Model Size: ", parameters:size()[1])
   parametersClone = parameters:clone()
end
print(model)
print(criterion)

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs
elseif opt.optimization == 'sgd' then
    optimState = {
      lr = opt.learningRate,
      lrd = opt.weightDecay,
      mom = opt.momentum
   }
   optimMethod = optim.msgd
elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      learningRateDecay = opt.learningRateDecay,
      momentum = opt.momentum,
      learningRateDecay = 0,
      dampening = 0,
      nesterov = opt.nesterov
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'RMSPROP' then
   optimState = {
      decay = opt.decayRMSProp,
      lr = opt.lrRMSProp,
      momentum = opt.momentumRMSProp,
      epsilon = opt.epsilonRMSProp
   }
   optimMethod = optim.rmspropsingle
else
   error('unknown optimization method')
end

function saveModel(s)
   torch.save(opt.outputprefix .. string.format("_%010.2f_model", s), parameters)
end

function loadModel(m)
   parameters:copy(torch.load(m))
end

function cleanMemForRuntime()
   parametersClone = nil
   gradParameters = nil
   model:get(1).gradWeight = nil
   model:get(3).gradWeight = nil 
   model:get(3).gradBias = nil
   model:get(6).gradWeight = nil
   model:get(6).gradBias = nil
   model:get(9).gradWeight = nil
   model:get(9).gradBias = nil
   model:get(11).gradWeight = nil
   model:get(11).gradBias = nil
   collectgarbage()
   collectgarbage()
end


function train()
    epoch = epoch or 1
    if optimState.evalCounter then
        optimState.evalCounter = optimState.evalCounter + 1
    end
--    optimState.learningRate = opt.learningRate
    local time = sys.clock()
    model:training()
    local batches = trainDataTensor:size()[1]/opt.batchSize
    local bs = opt.batchSize
    shuffle = torch.randperm(batches)
    for t = 1,batches,1 do
        local begin = (shuffle[t] - 1)*bs + 1
 --       local input = trainDataTensor:narrow(1, begin , bs) 
 --       local target = trainDataTensor_y:narrow(1, begin , bs)
        local input = opt.dataoncpu and trainDataTensor:narrow(1, begin , bs):cuda() or trainDataTensor:narrow(1, begin , bs)
        local target = opt.dataoncpu and trainDataTensor_y:narrow(1, begin , bs):cuda() or trainDataTensor_y:narrow(1, begin , bs)
        local input_lstm_fwd = trainDataTensor_lstm_fwd:narrow(1, begin , bs)
        local input_lstm_bwd = trainDataTensor_lstm_bwd:narrow(1, begin , bs)
       
        
        local feval = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            local f = 0
            if true then
               local output = model:forward(input)
               f = criterion:forward(output, target)
               local df_do = criterion:backward(output, target)
               model:backward(input, df_do)
            else
               local output = model:forward(input)
               f = criterion:forward(output, target)
               local df_do = criterion:backward(output, target)
               model:backward(input, df_do) 
            end
            --cutorch.synchronize()
            if opt.L1reg ~= 0 then
               local norm, sign = torch.norm, torch.sign
               f = f + opt.L1reg * norm(parameters,1)
               gradParameters:add( sign(parameters):mul(opt.L1reg) )
            end
            if opt.L2reg ~= 0 then
    --           local norm, sign = torch.norm, torch.sign
    --           f = f + opt.L2reg * norm(parameters,2)^2/2
               parametersClone:copy(parameters)
               gradParameters:add( parametersClone:mul(opt.L2reg) )
            end
            gradParameters:clamp(-opt.gradClip, opt.gradClip)
            return f,gradParameters
        end

        if optimMethod == optim.asgd then
            _,_,average = optimMethod(feval, parameters, optimState)
        else
--            a,b = model:parameters()
         --   print('a ' .. a[1][1][1]);
            optimMethod(feval, parameters, optimState)
         --   print('  ' .. a[1][1][1]);
        end
    end

    time = sys.clock() - time
    print("\n==> time for 1 epoch = " .. (time) .. ' seconds')
end

function test(inputDataTensor, inputDataTensor_lstm_fwd, inputDataTensor_lstm_bwd, inputTarget, state)
    local time = sys.clock()
    model:evaluate()
    local bs = opt.batchSizeTest
    local batches = inputDataTensor:size()[1]/bs
    local correct = 0
    local correct2 = 0
    local correct3 = 0
    local curr = -1
    for t = 1,batches,1 do
        curr = t
        local begin = (t - 1)*bs + 1
        --local input = inputDataTensor:narrow(1, begin , bs)
        local input = opt.dataoncpu and inputDataTensor:narrow(1, begin , bs):cuda() or inputDataTensor:narrow(1, begin , bs)
        local input_lstm_fwd = inputDataTensor_lstm_fwd:narrow(1, begin , bs)
        local input_lstm_bwd = inputDataTensor_lstm_bwd:narrow(1, begin , bs)
        local pred        
        pred = model:forward(input)
   
        local prob, pos
        if opt.twoCriterion then
           prob, pos = torch.max(pred[1], 2)
        else
           prob, pos = torch.max(pred, 2)
        end
        for m = 1,bs do
          for k,v in ipairs(inputTarget[begin+m-1]) do
            if pos[m][1] == v then
                correct = correct + 1
                break
            end
          end
          for k,v in ipairs(inputTarget[begin+m-1]) do
            if torch.abs(pos[m][1] - v) < 2 then
              correct2 = correct2 + 1
              break
            end
          end 
          for k,v in ipairs(inputTarget[begin+m-1]) do
            if torch.abs(pos[m][1] - v) < 3 then
              correct3 = correct3 + 1
              break
            end
          end
        end     
    end

    local rest_size = inputDataTensor:size()[1] - curr * bs
    if rest_size > 0 then
       local input
       local input_lstm_fwd
       local input_lstm_bwd
       if opt.type == 'cuda' then
          input = torch.CudaTensor(bs, inputDataTensor:size(2)):zero()
          input_lstm_fwd = torch.CudaTensor(bs, inputDataTensor_lstm_fwd:size(2)):zero()
          input_lstm_bwd = torch.CudaTensor(bs, inputDataTensor_lstm_bwd:size(2)):zero()
       else
          input = torch.FloatTensor(bs, inputDataTensor:size(2)):zero()
          input_lstm_fwd = torch.FloatTensor(bs, inputDataTensor_lstm_fwd:size(2)):zero()
          input_lstm_bwd = torch.FloatTensor(bs, inputDataTensor_lstm_bwd:size(2)):zero()
       end
       input:narrow(1,1,rest_size):copy(inputDataTensor:narrow(1, curr*bs + 1, rest_size))
       input_lstm_fwd:narrow(1,1,rest_size):copy(inputDataTensor_lstm_fwd:narrow(1, curr*bs + 1, rest_size))
       input_lstm_bwd:narrow(1,1,rest_size):copy(inputDataTensor_lstm_bwd:narrow(1, curr*bs + 1, rest_size))
       local pred
       pred = model:forward(input)

       local prob, pos 
       if opt.twoCriterion then
           prob, pos = torch.max(pred[1], 2)
       else
           prob, pos = torch.max(pred, 2)
       end
       for m = 1,rest_size do
          for k,v in ipairs(inputTarget[curr*bs+m]) do
            if pos[m][1] == v then
                correct = correct + 1
                break
            end
          end
          for k,v in ipairs(inputTarget[curr*bs+m]) do
            if torch.abs(pos[m][1] - v) < 2 then
                correct2 = correct2 + 1
                break
            end
          end
          for k,v in ipairs(inputTarget[curr*bs+m]) do
            if torch.abs(pos[m][1] - v) < 3 then
                correct3 = correct3 + 1
                break
            end
          end
       end
    end
     
    state.bestAccuracy = state.bestAccuracy or 0
    state.bestEpoch = state.bestEpoch or 0
    state.bestAccuracy2 = state.bestAccuracy2 or 0
    state.bestEpoch2 = state.bestEpoch2 or 0
    state.bestAccuracy3 = state.bestAccuracy3 or 0
    state.bestEpoch3 = state.bestEpoch3 or 0
    local currAccuracy = correct/(inputDataTensor:size()[1])
    local currAccuracy2 = correct2/(inputDataTensor:size()[1])
    local currAccuracy3 = correct3/(inputDataTensor:size()[1])
    if currAccuracy > state.bestAccuracy then state.bestAccuracy = currAccuracy; state.bestEpoch = epoch end
    if currAccuracy2 > state.bestAccuracy2 then state.bestAccuracy2 = currAccuracy2; state.bestEpoch2 = epoch end
    if currAccuracy3 > state.bestAccuracy3 then state.bestAccuracy3 = currAccuracy3; state.bestEpoch3 = epoch end
    print(string.format("Epoch %s Accuracy: %s, best Accuracy: %s on epoch %s at time %s", epoch, currAccuracy, state.bestAccuracy, state.bestEpoch, sys.toc() ))
    print(string.format("Epoch %s Accuracy2: %s, best Accuracy: %s on epoch %s at time %s", epoch, currAccuracy2, state.bestAccuracy2, state.bestEpoch2, sys.toc() ))
    print(string.format("Epoch %s Accuracy3: %s, best Accuracy: %s on epoch %s at time %s", epoch, currAccuracy3, state.bestAccuracy3, state.bestEpoch3, sys.toc() )) 
end

