function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% (See Section 2.3 of the lecture notes.)
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a
% time. 
epsilon = 10^(-4);
n = size(theta, 1);
% J1 = zeros(size(numgrad));
% J2 = zeros(size(numgrad));
J1 = zeros(1, 1);
J2 = zeros(1, 1);
grad = zeros(size(numgrad));
temp1 = zeros(size(theta));
temp2 = zeros(size(theta));

for i = 1 : n
%     i
    temp1 = theta;
    temp2 = theta;
    temp1(i) = temp1(i) + epsilon;
    temp2(i) = temp2(i) - epsilon;
    [J1, grad] = J(temp1);
    [J2, grad] = J(temp2);
    numgrad(i) = (J1 - J2) / (2*epsilon);
end


% theta1 = theta + epsilon;
% theta2 = theta - epsilon;
% [J1(i), grad] = J(temp1);
% [J2(i), grad] = J(temp2);




%% ---------------------------------------------------------------
end
