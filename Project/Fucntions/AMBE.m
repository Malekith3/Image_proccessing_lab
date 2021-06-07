function [AMBE] = AMBE(img1,img2)
%This function calculates AMBE between two images
E1=mean(img1,'all');
E2=mean(img2,'all');
AMBE=abs(E1-E2);

end
