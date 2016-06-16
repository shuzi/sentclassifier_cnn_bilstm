#Minwei Feng
local MapTable, parent = torch.class('nn.MapTable', 'nn.Container')

function MapTable:__init()
   parent.__init(self)
   self.modules = {}
   self.output = {}
   self.gradInput = {}
end

function MapTable:updateOutput(input)
   for i=1,#input do
      self.output[i] = self:rethrowErrors(self.modules[1], 1, 'updateOutput', input[i])
   end
   return self.output
end

function MapTable:updateGradInput(input, gradOutput)
   for i=1,#input do
      self.gradInput[i] = self:rethrowErrors(self.modules[1], 1, 'updateGradInput', input[i], gradOutput[i])
   end
   return self.gradInput
end

function MapTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   for i=1,#input do
      self:rethrowErrors(self.modules[1], 1, 'accGradParameters', input[i], gradOutput[i], scale)
   end
end

function MapTable:accUpdateGradParameters(input, gradOutput, lr)
   lr = lr or 1
   for i=1,#input do
      self:rethrowErrors(self.modules[1], 1, 'accUpdateGradParameters', input[i], gradOutput[i], lr)
   end
end

function MapTable:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = '  |`-> '
   local ext = '  |    '
   local extlast = '       '
   local last = '   ... -> '
   local str = torch.type(self)
   str = str .. ' {' .. line .. tab .. 'input'
   for i=1,#self.modules do
      if i == self.modules then
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
      else
         str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
      end
   end
   str = str .. line .. tab .. last .. 'output'
   str = str .. line .. '}'
   return str
end