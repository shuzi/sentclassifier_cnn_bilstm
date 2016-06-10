dofile('optim-rmsprop-single.lua')

L_lstm_fwd = nn.LookupTableMaskZero(mapWordIdx2Vector:size()[1], opt.embeddingDim)
L_lstm_bwd = nn.LookupTableMaskZero(mapWordIdx2Vector:size()[1], opt.embeddingDim)
L_lstm_fwd.weight:sub(2,-1):copy(mapWordIdx2Vector)
L_lstm_bwd.weight = L_lstm_fwd.weight
L_lstm_bwd.gradWeight = L_lstm_fwd.gradWeight
LSTM_fwd = nn.Sequencer(nn.FastLSTM(opt.embeddingDim, opt.LSTMhiddenSize):trimZero(1))
LSTM_bwd = nn.Sequencer(nn.FastLSTM(opt.embeddingDim, opt.LSTMhiddenSize):trimZero(1))


rnn_fwd = nn.Sequential()
rnn_fwd:add(L_lstm_fwd)
if opt.dropout > 0 then
  rnn_fwd:add(nn.Dropout(opt.dropout))
end
if opt.LSTMmode == 1 then
  rnn_fwd:add(nn.SplitTable(1,2))
  rnn_fwd:add(LSTM_fwd)
  rnn_fwd:add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
elseif opt.LSTMmode == 2 then
  rnn_fwd:add(nn.SplitTable(1,2))
  rnn_fwd:add(LSTM_fwd)
  rnn_fwd:add(nn.JoinTable(2))
  rnn_fwd:add(nn.View(opt.batchSize, trainDataTensor_ydim, opt.LSTMhiddenSize))
  rnn_fwd:add(nn.AddConstantNeg(-20000))
  rnn_fwd:add(nn.Max(2))
elseif opt.LSTMmode == 3 then
  lstm_fwd = nn.SeqLSTM(opt.embeddingDim, opt.LSTMhiddenSize)
  lstm_fwd.batchfirst = true
  lstm_fwd.maskzero = true
  rnn_fwd:add(lstm_fwd)
  rnn_fwd:add(nn.AddConstantNeg(-20000))
  rnn_fwd:add(nn.Max(2))
elseif opt.LSTMmode == 4 then
  lstm_fwd = nn.SeqLSTM(opt.embeddingDim, opt.LSTMhiddenSize)
  lstm_fwd.batchfirst = true
  lstm_fwd.maskzero = true
  rnn_fwd:add(lstm_fwd)
  rnn_fwd:add(nn.Select(2,-1))
elseif opt.LSTMmode == 5 then
  lstm_fwd = cudnn.LSTM(opt.embeddingDim, opt.LSTMhiddenSize, 1, true)
  rnn_fwd:add(lstm_fwd)
  if opt.useACN then
    rnn_fwd:add(nn.AddConstantNeg(-20000))
  end
  rnn_fwd:add(nn.Max(2))
elseif opt.LSTMmode == 6 then
  lstm_fwd = cudnn.LSTM(opt.embeddingDim, opt.LSTMhiddenSize, 1, true)
  rnn_fwd:add(lstm_fwd)
  rnn_fwd:add(nn.Select(2,-1))
end

rnn_bwd = nn.Sequential()
rnn_bwd:add(L_lstm_bwd)
if opt.dropout > 0 then
  rnn_bwd:add(nn.Dropout(opt.dropout))
end
if opt.LSTMmode == 1 then
  rnn_bwd:add(nn.SplitTable(1,2))
  rnn_bwd:add(LSTM_bwd)
  rnn_bwd:add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
elseif opt.LSTMmode == 2 then
  rnn_bwd:add(nn.SplitTable(1,2))
  rnn_bwd:add(LSTM_bwd)
  rnn_bwd:add(nn.JoinTable(2))
  rnn_bwd:add(nn.View(opt.batchSize, trainDataTensor_ydim, opt.LSTMhiddenSize))
  rnn_bwd:add(nn.AddConstantNeg(-20000))
  rnn_bwd:add(nn.Max(2))
elseif opt.LSTMmode == 3 then
  lstm_bwd = nn.SeqLSTM(opt.embeddingDim, opt.LSTMhiddenSize)
  lstm_bwd.batchfirst = true
  lstm_bwd.maskzero = true
  rnn_bwd:add(lstm_bwd)
  rnn_bwd:add(nn.AddConstantNeg(-20000))
  rnn_bwd:add(nn.Max(2))
elseif opt.LSTMmode == 4 then
  lstm_bwd = nn.SeqLSTM(opt.embeddingDim, opt.LSTMhiddenSize)
  lstm_bwd.batchfirst = true
  lstm_bwd.maskzero = true
  rnn_bwd:add(lstm_bwd)
  rnn_bwd:add(nn.Select(2,-1))
elseif opt.LSTMmode == 5 then
  lstm_bwd = cudnn.LSTM(opt.embeddingDim, opt.LSTMhiddenSize, 1, true)
  rnn_bwd:add(lstm_bwd)
  if opt.useACN then
    rnn_bwd:add(nn.AddConstantNeg(-20000))
  end
  rnn_bwd:add(nn.Max(2))
elseif opt.LSTMmode == 6 then
  lstm_bwd = cudnn.LSTM(opt.embeddingDim, opt.LSTMhiddenSize, 1, true)
  rnn_bwd:add(lstm_bwd)
  rnn_bwd:add(nn.Select(2,-1))
end

model = nn.Sequential()
if opt.LSTMmode == 7 then
  bilstm = cudnn.BLSTM(opt.embeddingDim, opt.LSTMhiddenSize, 1, true)
  model:add(L_lstm_fwd)
  model:add(bilstm)
  if opt.useACN then
    model:add(nn.AddConstantNeg(-20000))
  end
  model:add(nn.Max(2))
else
  bilstm = nn.ParallelTable()
  bilstm:add(rnn_fwd):add(rnn_bwd)
  model:add(bilstm)
  model:add(nn.JoinTable(2))
end

--model:add(nn.Dropout(0.5))
--model:add(cudnn.BatchNormalization(opt.hiddenDim + 2*opt.LSTMhiddenSize))
model:add(nn.Linear(2*opt.LSTMhiddenSize, 2*opt.LSTMhiddenSize))
if opt.lastReLU then
  model:add(nn.ReLU())
else
  model:add(nn.Tanh())
end
model:add(nn.Linear(2*opt.LSTMhiddenSize, opt.numLabels))
model:add(nn.LogSoftMax())

if opt.twoCriterion then
  prob_idx = nn.ConcatTable()
  prob_idx:add(nn.Identity())
  prob_idx:add(nn.ArgMax(2,opt.numLabels, false))
  model:add(prob_idx)

  nll = nn.ClassNLLCriterion()
  abs = nn.AbsCriterion()
  criterion = nn.ParallelCriterion(true):add(nll, opt.criterionWeight):add(abs)
else
  criterion = nn.ClassNLLCriterion()
end


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
      mom = opt.momentum,
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
        local input = trainDataTensor:narrow(1, begin , bs) 
        local target = trainDataTensor_y:narrow(1, begin , bs)
        local input_lstm_fwd = trainDataTensor_lstm_fwd:narrow(1, begin , bs)
        local input_lstm_bwd = trainDataTensor_lstm_bwd:narrow(1, begin , bs)
        
        local feval = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            local f = 0
            if opt.LSTMmode == 7 then
               local output = model:forward(input_lstm_fwd)
               f = criterion:forward(output, target)
               local df_do = criterion:backward(output, target)
               model:backward(input_lstm_fwd, df_do)
            else 
               local output = model:forward{input_lstm_fwd, input_lstm_bwd}
               f = criterion:forward(output, target)
               local df_do = criterion:backward(output, target)
               model:backward({input_lstm_fwd, input_lstm_bwd}, df_do) 
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
--           gradParameters:clamp(-opt.gradClip, opt.gradClip)
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
    local curr = -1
    for t = 1,batches,1 do
        curr = t
        local begin = (t - 1)*bs + 1
        local input = inputDataTensor:narrow(1, begin , bs)
        local input_lstm_fwd = inputDataTensor_lstm_fwd:narrow(1, begin , bs)
        local input_lstm_bwd = inputDataTensor_lstm_bwd:narrow(1, begin , bs)

        local pred
        if opt.LSTMmode == 7 then
           pred = model:forward(input_lstm_fwd)
        else 
           pred = model:forward{input_lstm_fwd, input_lstm_bwd}
        end
        
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
       if opt.LSTMmode == 7 then
           pred = model:forward(input_lstm_fwd)
       else
           pred = model:forward{input_lstm_fwd, input_lstm_bwd}
       end

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
       end
    end
     
    state.bestAccuracy = state.bestAccuracy or 0
    state.bestEpoch = state.bestEpoch or 0
    state.bestAccuracy2 = state.bestAccuracy2 or 0
    state.bestEpoch2 = state.bestEpoch2 or 0
    local currAccuracy = correct/(inputDataTensor:size()[1])
    local currAccuracy2 = correct2/(inputDataTensor:size()[1])
    if currAccuracy > state.bestAccuracy then state.bestAccuracy = currAccuracy; state.bestEpoch = epoch end
    if currAccuracy2 > state.bestAccuracy2 then state.bestAccuracy2 = currAccuracy2; state.bestEpoch2 = epoch end
    print(string.format("Epoch %s Accuracy: %s, best Accuracy: %s on epoch %s at time %s", epoch, currAccuracy, state.bestAccuracy, state.bestEpoch, sys.toc() ))
    print(string.format("Epoch %s Accuracy2: %s, best Accuracy: %s on epoch %s at time %s", epoch, currAccuracy2, state.bestAccuracy2, state.bestEpoch2, sys.toc() ))
end

