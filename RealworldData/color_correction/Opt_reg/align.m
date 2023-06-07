function [ImTrans,tau] = align(Im,Imref,tau,iter)

c=size(Im,3);
ImTrans = Im;
if nargin<3
  tau = zeros(6,1);
end
if nargin<4
  iter = 5;
end

numLevel = fix(log(size(Im,1)*size(Im,2))/log(2)/2-3)-1;

I1=mean(Imref,3);
I2=mean(Im,3);
% I2=luminance_transfer(I1,I2);
[~,tau] = regMGNC(I1,I2,tau,numLevel,iter);

ImTrans=warpImg(Im,tau);


end
