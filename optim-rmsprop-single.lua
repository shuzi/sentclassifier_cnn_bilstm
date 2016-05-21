require 'optim'

function optim.rmspropsingle(opfunc, w, config, state)
  local config = config or {}
  local state = state or config
  local decay = config.decay
  local lr = config.lr
  local momentum = config.momentum
  local epsilon = config.epsilon


  state.gradAccum = state.gradAccum or torch.CudaTensor():resizeAs(w):fill(0)
  state.gradSqAccum = state.gradSqAccum or torch.CudaTensor():resizeAs(w):fill(0)
  state.update = state.update or torch.CudaTensor():resizeAs(w):fill(0)
  state.gradRms = state.gradRms or torch.CudaTensor():resizeAs(w):fill(0)

  local fx,dfdx = opfunc(w)

  state.gradAccum:mul(decay):add(1 - decay, dfdx)
  state.gradSqAccum:mul(decay):add(1 - decay, torch.cmul(dfdx, dfdx))
  state.gradRms:copy(state.gradSqAccum)
    :add(-1, torch.cmul(state.gradAccum, state.gradAccum))
    :add(epsilon):sqrt()
  state.update:mul(momentum):add(-lr, torch.cdiv(dfdx, state.gradRms))
  
  w:add(state.update)

  return w,{fx}
end
