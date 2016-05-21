local AddConstantNeg, parent = torch.class('nn.AddConstantNeg', 'nn.Module')

function AddConstantNeg:__init(constant_scalar, ip)
   parent.__init(self)
   assert(type(constant_scalar) == 'number', 'input is not scalar!')
   self.constant_scalar = constant_scalar
   
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
end

function AddConstantNeg:updateOutput(input)
  if self.inplace then
    input:add(self.constant_scalar)
    self.output:set(input)
    for i=1,input:size(1) do
       for j=1,input:size(2) do
          if self.output[i][j]:norm(1,1) == 0 then
            self.output[i][j]=self.constant_scalar
          end
       end
    end 
  else
    self.output:resizeAs(input)
    self.output:copy(input)
    for i=1,input:size(1) do
       for j=1,input:size(2) do
          if (self.output[i][j]):norm(1,1)[1] == 0 then
            self.output[i][j]=self.constant_scalar
          end
       end
    end
  end
  return self.output
end 

function AddConstantNeg:updateGradInput(input, gradOutput)
  if self.inplace then
    self.gradInput:set(gradOutput)
    -- restore previous input value
    input:add(-self.constant_scalar)
  else
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
  end
  return self.gradInput
end

