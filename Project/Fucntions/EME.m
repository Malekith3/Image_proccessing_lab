function [EME] = EME(img)
%This function calculates EME of an image
epsilon=1e-12;

if(size(img,2)==3)
    img=rgb2gray(img);
end

[rows cols]=size(img);
k1=round(rows/8);
k2=round(cols/8);
Imax=zeros(k1,k2);
Imin=zeros(k1,k2);

for ii=1:k1
    for jj=1:k2
        Bij=img(ii:ii+8,jj:jj+8);
        Imax(ii,jj)=max(Bij,[],'all');
        Imin(ii,jj)=min(Bij,[],'all');
    end
end

EME=(1/(k1*k2))*sum(20*log(Imax./(Imin+epsilon)),'all');
end