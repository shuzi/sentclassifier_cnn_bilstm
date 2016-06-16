-- From https://github.com/fmassa/torch-nn/blob/master/TopK.lua

local TopK, parent = torch.class('nn.TopK', 'nn.Module')

function TopK:__init(k, dimension, dir, sort)
  parent.__init(self)
  self.k = k or 1
  self.dimension = dimension or 1
  self.dir = dir or false
  self.sort = sort or false
end

function TopK:_lazyInit()
  self._indices = self._indices or
  (torch.type(self.output) == 'torch.CudaTensor' and torch.CudaTensor() or torch.LongTensor())
end

function TopK:updateOutput(input)
  self:_lazyInit()
  local dimension = self.dimension
  local k = self.k
  torch.topk(self.output, self._indices, input, k, dimension, self.dir, self.sort)
  return self.output
end

function TopK:updateGradInput(input, gradOutput)
  self:_lazyInit()
  local dimension = self.dimension
  self.gradInput:resizeAs(input):zero():scatter(dimension, self._indices, gradOutput)
  return self.gradInput
end

function TopK:type(type, tensorCache)
  -- torch.max expects a LongTensor as indices, whereas cutorch.max expects a CudaTensor.
  if type == 'torch.CudaTensor' then
    parent.type(self, type, tensorCache)
  else
    -- self._indices must be a LongTensor. Setting it to nil temporarily avoids
    -- unnecessary memory allocations.
    local indices
    indices, self._indices = self._indices, nil
    parent.type(self, type, tensorCache)
    self._indices = indices and indices:long() or nil
  end
  return self
end

function TopK:clearState()
  nn.utils.clear(self, '_indices')
  return parent.clearState(self)
end