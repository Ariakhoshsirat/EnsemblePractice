function [ M ] = Fivefolds( data )
%TENFOLD Summary of this function goes here
%   Detailed explanation goes here

    n = size(data,1);
    A = randperm(n);
    start = 1 ;
    step = floor(n/5);
    M = zeros(step,5);
    
    for i = 1:5
      M(:,i) = A(start:start+step-1);
      start = start + step;
    end



end

