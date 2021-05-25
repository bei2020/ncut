%===== binar1.m
fs=dir('photos');
%buttg=imread('photos/img_003.bmp');
buttg=imread(strcat('photos/',fs(4).name));
%butt=buttg(40:59, 10:49, :);
butt=buttg(110:200,140:300,1);
[nlig ncol]=size(butt); nbpix=prod(size(butt));
%===== global histogram
pixc3=zeros(nlig*ncol,1); pixc3(:)=butt;
histog=hist(pixc3,256)/nlig/ncol;
figure(1); plot([0:255] ,histog); grid
%===== thresholds based on a visual examination
% of the histogram
figure(2); subplot(131);
imagesc(butt); axis('image'); colormap(gray);
%===== threshold 1
pixc2=zeros(nlig,ncol);
seuil=152; idxy=find(butt>seuil);
pixc2(idxy)=255*ones(size(idxy)); subplot(132); imagesc(pixc2);
axis('image'); colormap(gray)
%===== threshold 2
pixc2=zeros(nlig,ncol);
seuil=90; idxy=find(butt>seuil);
pixc2(idxy)=255*ones(size(idxy)); subplot(133); imagesc(pixc2);
axis('image'); colormap(gray)
save histog histog