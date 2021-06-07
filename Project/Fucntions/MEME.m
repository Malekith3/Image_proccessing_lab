function [MEME] = MEME(img)
%this function calculates MEME according to the paper
alpha=100;
if(size(img,2)==3)
    img=rgb2gray(img);
end
[rows cols]=size(img);
k1=round(rows/8);
k2=round(cols/8);
IDC=zeros(k1,k2);
Cij=zeros(k1,k2);
for ii=1:k1
    for jj=1:k2
        Bij=img(ii:ii+8,jj:jj+8);       %block
        Cij(ii,jj)=std(double(Bij),1,'all');
        IDC(ii,jj)=mean(Bij,'all');
    end
end

CIDC=std(double(IDC),1,'all');

MEME=alpha*(CIDC/(k1*k2))*mean(Cij,'all');

end