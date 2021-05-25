%===== binarOtsu .m
fs=dir('photos');
%buttg=imread('photos/img_003.bmp');
%buttg=imread(strcat('photos/',fs(4).name));
%butt=buttg(40:59, 10:49, :);
%pixc=buttg(:,:,1);%(110:200,140:300,1);
pixc=cast(trip*255,'uint8');
nlig=size(pixc,1); ncol=size(pixc,2);
figure(1); colormap('gray');
subplot(121) ; imagesc(pixc); axis('image')
%=====
[threshold,Otsu]=otsu(pixc);
pixc2=255*(pixc>threshold) ;%pixc2=zeros(nlig,ncol);
subplot(122) ; imagesc(pixc2);
axis('image'); colormap(gray)