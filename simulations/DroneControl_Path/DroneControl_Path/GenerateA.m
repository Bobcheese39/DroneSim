% This script generate (x,y,z) data points by writing the equations.
Tfinal = 33;
tss = 0.1;
t = 0:tss:Tfinal;

% Write the equations of your path
x = 2*sin(1*t);
y = 2 - 2*cos(1*t);
z = 3 + 1*sin(2*t);

A = zeros(length(t),3);
 for i = 1:length(x)
     A(i,1) = x(i)
     A(i,2) = y(i)
     A(i,3) = z(i)
 end
 
 clear x;
 clear y;
 clear tss;
 clear t;