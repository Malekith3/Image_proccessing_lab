function [enhancedImage] = ContrastEnhancement(img)
%This function excecutes the contrast enhancement according to the given paper
%h_m (k)=c(h_i (k)+δ)^γ)
%
h= imhist(img);
delta=std(h,1);
mu=mean(img,'all');

if mu<128               
    gamma =(255-mu)/255;
else
    gamma = mu/255;
end

hm=(h+delta).^gamma;
enhancedImage = histeq(img,hm);
end


