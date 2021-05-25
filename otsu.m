function [threshold,Hs]=otsu(pixc)
%!======================================================!
%! Binarization with the Otsu method !
%! SYNOPSIS: [threshold,Hs)=OTSU(pixc)
%! pixc image with 256 gray levels
%! threshold = calculated optimal threshold
%! Hs = criterium to be maximized for s=0:255
%!======================================================!
%===== histogram of the image
[nlig ncol]=size(pixc) ;
Lhist=256; histog=zeros(Lhist,1);
for k=1:Lhist
histog(k)=length(find(pixc==k-1));
end
histog=histog/nlig/ncol;
%===== calculating the criterium
Pinf=0; sinf=0;% Psup=l; muinf=0 ;
ssup=(0:Lhist-1)*histog ;
%===== calculating the criterium for the values of S
% Hs = Pinf*Psup*(muinf-musup)*(muinf-musup)
Hs=zeros(1,Lhist);
for S=0:Lhist-2
%===== distributions
Pinf=Pinf+histog(S+1); Psup=1-Pinf;
%===== local means
sinf=sinf+S*histog(S+1); ssup=ssup-S*histog(S+1);
muinf=sinf/Pinf; musup=ssup/Psup;
Hs(S+1)=Pinf*Psup*(muinf-musup)*(muinf-musup);
end
threshold=find(Hs==max(Hs)); % threshold
return