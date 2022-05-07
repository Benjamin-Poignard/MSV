function [xold,fold,invhess,para]=optint(xnew,fnew,para)
%OPTINT Function to initialize FMINU routine.

%	Copyright (c) 1990 by the MathWorks, Inc.
%	Andy Grace 7-9-90.
lenx=length(xnew);
invhess=eye(lenx);  
xold=xnew;
fold=fnew;
para=foptions(para);
if para(14)==0, para(14)=lenx*100;end 
if para(1)>1, para, end
if para(1)>0,
	disp('')
	disp('f-COUNT   FUNCTION    STEP-SIZE      GRAD/SD  LINE-SEARCH')
end