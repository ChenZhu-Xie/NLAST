function [A3,profilex,profiley,d_k]= SSF_FD(A,L,m,n,lam1,lam2,p,T,A1,A2,structure_path,md)
%分步傅里叶算法+有限差分法
%输出晶体后端面光场A3
kx = (-m/2+0.5:m/2-0.5).*(2*pi/A);    %计算频域坐标
ky = (-m/2+0.5:m/2-0.5).*(2*pi/A);
[Kx,Ky] = meshgrid(kx,ky);
d_step = L/n;    %计算纵向间隔
lam3 = 1/(1/lam1+1/lam2);

%计算纵向相位失配量
switch p(1)
    case 'e'
        n1 = C_n_LN(lam1,T,'e');
    case 'o'
        n1 = C_n_LN(lam1,T,'o');
end
switch p(2)
    case 'e'
        n2 = C_n_LN(lam2,T,'e');
    case 'o'
        n2 = C_n_LN(lam2,T,'o');
end
switch p(3)
    case 'e'
        n3 = C_n_LN(lam3,T,'e');
    case 'o'
        n3 = C_n_LN(lam3,T,'o');
end
k1 = n1*2*pi/lam1;
k2 = n2*2*pi/lam2;
k3 = n3*2*pi/lam3;
d_k = k1+k2-k3;
co = 4-2*(lam1==lam2);%计算非线性系数系数

%一些初始参数
deff = 22e-6;c = 3e14;
w3 = 2*pi*c/lam3;

A3 = zeros(m);profilex = zeros(m,n);profiley = zeros(m,n);
H = fftshift(exp(-1i/k3*(Kx.^2+Ky.^2)*d_step));

%根据分步傅里叶和有限差分法步进计算A3,每前进一步读取当前结构参数
for ii = 1:n
    load([structure_path '\Structure_face' num2str(ii) '.mat']);
    z = (ii-1)*d_step;
    d_A3_linear = ifft2(fft2(A3).*H);
    d_A3_nonlinear = (w3^2/(k3*c^2))*exp(1i*d_k*z).*((exp(1i*d_k*d_step)-1)/d_k).*A1.*A2.*(co*deff).*(1-md*face);
    A3 = d_A3_linear + d_A3_nonlinear;
    profilex(:,ii) = A3(fix(m/2+1),:);
    profiley(:,ii) = A3(:,fix(m/2+1));
    clc;disp(ii/n);
end
%结束，输出晶体后端面A3、A3在晶体中产生传播的x侧面图、A3在晶体中产生传播的y侧面图
end

