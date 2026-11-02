clear
close all
clc

Q=[0.25,0.5,0.75];
alpha=[0,0.3,0.5,0.7,1];
ibselon_vec=[0.05,0.25];       
Nib=numel(ibselon_vec);      

% clear
% close all
% clc

%————Parameters———%
%channel model, Friis transmission equation
T=[0,0];                      %location of hybrid transmitter
Wil=[4,3];                    %location of Willie
d_tw=sqrt((T(1)-Wil(1))^2+(T(2)-Wil(2))^2);
Gtw=2;    %定向天线
c=3*10^8;
f=915*10^6;
K=c/(f*4*pi);
r=(Gtw*K^2)/(d_tw^2);          % square chanel gain |htw|2
R_optimal_recorder_1=zeros(251,3,5,Nib);   
R_optimal_recorder_2=zeros(251,3,5,Nib);   
% r=1;
pt_sol=zeros(251,3,5,Nib);                
pt_sol_2=zeros(251,3,5,Nib);               
max_throughput=zeros(3,5,Nib);             
idex_m=zeros(3,5,Nib);                    
max_throughput_2=zeros(3,5,Nib);          
idex_m_2=zeros(3,5,Nib);                   



sigmman2=0.0001;   %noise variances
G=1;                           %the energy beamforming gain
delta=0.25;                    %the power reflection coefficient
eta=0.65;                      %the efficiency of RF-to-DC energy conversion
w=0.5;                         %the time fraction for harvesting energy in active RF transmission
rho_A=113*10^(-6);             %the circuit power consumption rate (in Watt) in active transmission 113uW
rho_B=8.9*10^(-6);             %the circuit power consumption rate (in Watt) in backscatter transmission 8.9uW
Pamax=10;
Pamin=0.001;
% pt_kongji=zeros(251,21,4);
% ibselon=0.25;                
k=0.2;
for eb=1:Nib                   
ibselon=ibselon_vec(eb);       
for e=1:5
    for j=1:3
        PrH0=1-Q(j);
        PrH1=Q(j)*alpha(e);
        PrH2=Q(j)*(1-alpha(e));
        i=0;
        last_value=0;
        last_value_2=0;
        for N=150:1:400
            i=i+1;
            b=Pamax;a=Pamin;tol=0.01;
            error_1=0;
            while (b - a) > tol   %这里使用了二分法求最优的Pt解
                %H1下的接收信号功率
                P1R_a=P1R_calc(a);
                P1R_b=P1R_calc(b);
                %H2下的接收信号功率
                P2R_a=P2R_calc(a);
                P2R_b=P2R_calc(b);
                %策略一的两个判决门限
                taovalue_a=double(solve_tao(a,N,PrH0,PrH1,PrH2));%发射功率为a时的判决门限
                taovalue_b=double(solve_tao(b,N,PrH0,PrH1,PrH2));%发射功率为b时的判决门限
                tao2value_a=double(solve_tao2(a,N,PrH1,PrH2));%发射功率为a时的判决门限
                tao2value_b=double(solve_tao2(b,N,PrH1,PrH2));%发射功率为b时的判决门限
                %有无传输检错概率1
                Pe1_a=P_error_I(P1R_a,P2R_a,taovalue_a,N,ibselon,PrH0,PrH1,PrH2);
                Pe1_b=P_error_I(P1R_b,P2R_b,taovalue_b,N,ibselon,PrH0,PrH1,PrH2);
                Pe2_a=P_error_II(P1R_a,P2R_a,taovalue_a,tao2value_a,N,PrH1,PrH2,PrH0);
                Pe2_b=P_error_II(P1R_b,P2R_b,taovalue_b,tao2value_b,N,PrH1,PrH2,PrH0);
                omega_a=Pe1_a+Pe2_a;
                omega_b=Pe1_b+Pe2_b;
                if omega_b*omega_a>0
                    error_1=1;
                    break  %两边同号，无法继续二分，错误
                end
                c=(a + b)/2;
                %二分法的中点重新计算功率和检错概率
                P1R_c= P1R_calc(c);    %H1下的接收信号功率
                P2R_c= P2R_calc(c);    %H1下的接收信号功率
                %策略1的c门限及积分区间
                taovalue_c=double(solve_tao(c,N,PrH0,PrH1,PrH2));
                taovalue2_c=double(solve_tao2(c,N,PrH1,PrH2));

                Pe1_c=P_error_I(P1R_c,P2R_c,taovalue_c,N,ibselon,PrH0,PrH1,PrH2);%有无传输检错概率1
                Pe2_c=P_error_II(P1R_c,P2R_c,taovalue_c,taovalue2_c,N,PrH1,PrH2,PrH0);
                omega_c=Pe1_c+Pe2_c;
                if abs(omega_c)<0.005
                    break
                elseif omega_c < 0
                    b = c;
                else
                    a = c;
                end
            end

            b2=Pamax;a2=Pamin;tol=0.01;
            error_2=0;
            while (b2 - a2) > tol  %这里使用了二分法求最优的Pt解
                %计算a,b下的功率P1R，P2R
                P1R_a2=P1R_calc(a2);    %H1下的接收信号功率
                P1R_b2=P1R_calc(b2);
                P2R_a2=P2R_calc(a2);    %H2下的接收信号功率
                P2R_b2=P2R_calc(b2);
                %策略二的两个判决门限
                [Fi_1_0_a2,Fi_2_1_a2,Fi_0_1_a2,Fi_1_2_a2,Fi_0_2_a2,Fi_2_0_a2]=Fi_a_b_calc(P1R_a2,P2R_a2,N,PrH0,PrH1,PrH2);
                [Fi_1_0_b2,Fi_2_1_b2,Fi_0_1_b2,Fi_1_2_b2,Fi_0_2_b2,Fi_2_0_b2]=Fi_a_b_calc(P1R_b2,P2R_b2,N,PrH0,PrH1,PrH2);
                %策略二积分区间
                if Fi_0_1_a2>Fi_1_2_a2 %有空集
                    d_a2=Fi_2_0_a2/sigmman2;
                    e_a_12=Fi_0_2_a2/P1R_a2;
                    e_a_22=Fi_0_2_a2/P2R_a2;
                    error_2_a2=gammainc(d_a2,N,'upper');  %H0看成1和2
                    error_3_a2=gammainc(e_a_12,N);  %(D0|H1)
                    error_4_a2=gammainc(e_a_22,N);  %(D0|H2)
                    error_5_a2=gammainc(e_a_12,N,'upper');  %(D2|H1)
                    error_6_a2=0;

                else
                    e_a_12=Fi_0_1_a2/P1R_a2;
                    e_a_22=Fi_0_1_a2/P2R_a2;
                    g_a2=Fi_1_0_a2/sigmman2;
                    error_2_a2=gammainc(g_a2,N,'upper');  %H0看成1和2
                    error_3_a2=gammainc(e_a_12,N);  %H1看成0
                    error_4_a2=gammainc(e_a_22,N);  %H2看成0
                    error_5_a2=gammainc(Fi_2_1_a2/P1R_a2,N,'upper');  %H1看成H2
                    error_6_a2=gammainc(Fi_1_0_a2/P2R_a2,N,'upper')-gammainc(Fi_1_2_a2/P2R_a2,N,'upper');%2看成1

                end
                if Fi_0_1_b2>Fi_1_2_b2  %有空集
                    d_b2=Fi_2_0_b2/sigmman2;
                    e_b_12=Fi_0_2_b2/P1R_b2;
                    e_b_22=Fi_0_2_b2/P2R_b2;
                    error_2_b2=gammainc(d_b2,N,'upper');
                    error_3_b2=gammainc(e_b_12,N);
                    error_4_b2=gammainc(e_b_22,N);
                    error_5_b2=gammainc(e_b_12,N,'upper');  %(D2|H1)
                    error_6_b2=0;

                else
                    e_b_12=Fi_0_1_b2/P1R_b2;
                    e_b_22=Fi_0_1_b2/P2R_b2;
                    g_b2=Fi_1_0_b2/sigmman2;
                    error_2_b2=gammainc(g_b2,N,'upper');
                    error_3_b2=gammainc(e_b_12,N);
                    error_4_b2=gammainc(e_b_22,N);
                    error_5_b2=gammainc(Fi_2_1_b2/P1R_b2,N,'upper');  %H1看成H2
                    error_6_b2=gammainc(Fi_1_0_b2/P2R_b2,N,'upper')-gammainc(Fi_1_2_b2/P2R_b2,N,'upper');%2看成1

                end
                if alpha(e)*alpha(e)==alpha(e)
                    error_6_b2=0;
                    error_5_b2=0;
                    error_6_a2=0;
                    error_5_a2=0;
                end
                %策略二检错概率
                omega_2_a2=double(1/(1-max(PrH0,max(PrH1,PrH2)))*(PrH0*error_2_a2+PrH1*(error_3_a2+error_5_a2)+PrH2*(error_4_a2+error_6_a2))-1+ibselon);
                omega_2_b2=double(1/(1-max(PrH0,max(PrH1,PrH2)))*(PrH0*error_2_b2+PrH1*(error_3_b2+error_5_b2)+PrH2*(error_4_b2+error_6_b2))-1+ibselon);
                if omega_2_a2*omega_2_b2>0
                    error_2=1;

                    break  %两边同号，无法继续二分，错误
                end

                c2=(a2 + b2)/2;
                %二分法的中点重新计算功率和检错概率
                P1R_c2=P1R_calc(c2);    %H1下的接收信号功率
                P2R_c2=P2R_calc(c2);    %H1下的接收信号功率
                %策略2的c门限及积分区间
                [Fi_1_0_c2,Fi_2_1_c2,Fi_0_1_c2,Fi_1_2_c2,Fi_0_2_c2,Fi_2_0_c2]=Fi_a_b_calc(P1R_c2,P2R_c2,N,PrH0,PrH1,PrH2);
                %策略2的c积分区间
                if Fi_1_0_c2>Fi_2_1_c2
                    d_c2=Fi_2_0_c2/sigmman2;
                    e_c_12=Fi_0_2_c2/P1R_c2;
                    e_c_22=Fi_0_2_c2/P2R_c2;
                    error_2_c2=gammainc(d_c2,N,'upper');
                    error_3_c2=gammainc(e_c_12,N);
                    error_4_c2=gammainc(e_c_22,N);
                    error_5_c2=gammainc(e_c_12,N,'upper');  %(D2|H1)
                    error_6_c2=0;%2看成1
                else
                    e_c_12=Fi_0_1_c2/P1R_c2;
                    e_c_22=Fi_0_1_c2/P2R_c2;
                    g_c2=Fi_1_0_c2/sigmman2;
                    error_2_c2=gammainc(g_c2,N,'upper');
                    error_3_c2=gammainc(e_c_12,N);
                    error_4_c2=gammainc(e_c_22,N);
                    error_5_c2=gammainc(Fi_2_1_c2/P1R_c2,N,'upper');  %H1看成H2
                    error_6_c2=gammainc(Fi_1_0_c2/P2R_c2,N,'upper')-gammainc(Fi_1_2_c2/P2R_c2,N,'upper');%2看成1
                end
                if alpha(e)*alpha(e)==alpha(e)
                    error_6_c2=0;
                    error_5_c2=0;
                end
                omega_2_c2=double(1/(1-max(PrH0,max(PrH1,PrH2)))*(PrH0*error_2_c2+PrH1*(error_3_c2+error_5_c2)+PrH2*(error_4_c2+error_6_c2))-1+ibselon);
                if abs(omega_2_c2)<0.005
                    break
                elseif omega_2_c2 < 0
                    b2 = c2;
                else
                    a2= c2;
                end

            end
            % if (a2+b2)/2<=pt_kongji(i,j,e)
            %     kongji_sign(i,j,e)=1;
            % end



            %策略1的最优吞吐量
            pt_sol(i,j,e,eb)= (a + b)/2;%二分法解出的pt
            if error_1==1
                pt_sol(i,j,e,eb)=1;
            end
            %计算策略1下的最终功率
            P1R_end=P1R_calc_bob(pt_sol(i,j,e,eb));    %H1下的接收信号功率
            P2R_end=P2R_calc_bob(pt_sol(i,j,e,eb));    %H2下的接收信号功率
            %策略1下的信噪比
            r_1=P1R_end/sigmman2-1;
            r_2=P2R_end/sigmman2-1;
            %寻找0.1到1.9范围内使得有效吞吐量最大的R
            g_1=@(R) ((r_1+1)*log(2)*sqrt(N)*(log2(1+r_1)-R))/sqrt(r_1^2+2*r_1);
            g_2=@(R) ((r_2+1)*log(2)*sqrt(N)*(log2(1+r_2)-R))/sqrt(r_2^2+2*r_2);
            throughput11=@(R) -(1-0.5*erfc(g_1(R)/sqrt(2)))*N*R;
            throughput22=@(R) -(1-0.5*erfc(g_2(R)/sqrt(2)))*N*R;
            % [R1_opt,~]=fminbnd(throughput11,0,1);
            % [R2_opt,~]=fminbnd(throughput22,0,1);
            R1_opt=findmin(throughput11,0,0.5);
            R2_opt=findmin(throughput22,0,0.5);
            R_optimal_recorder_1(i,j,e,eb)=R1_opt;   %【修改1】下标增加 eb
            R_optimal_recorder_2(i,j,e,eb)=R2_opt;   %【修改1】下标增加 eb

            %寻找k约束下的最大R的范围，并取该值与优化后的R的最小值
            code_error1=@(R) double(0.5*erfc(g_1(R)/sqrt(2))-k);
            code_error2=@(R) double(0.5*erfc(g_2(R)/sqrt(2))-k);
            % R1_max=bisection_method(code_error1,0,4,0.0001,e);
            % R2_max=bisection_method(code_error2,0,4,0.0001,e);%通过二分法寻找k约束下的R的最大值
            R1_max=findRmax(code_error1,0,2);
            R2_max=findRmax(code_error2,0,2);
            R1_end=min(R1_max,R1_opt);
            R2_end=min(R2_max,R2_opt);
            % if R1_max(i,j,e)==R1_end
            %     R1_max_indicator(i,j,e)=1;
            % end
            % if R2_max(i,j,e)==R1_end
            %     R2_max_indicator(i,j,e)=1;
            % end
            %得到最终的有效吞吐量
            g_1_end=((r_1+1)*log(2)*sqrt(N)*(log2(1+r_1)-R1_end))/sqrt(r_1^2+2*r_1);%Q函数内部的x
            g_2_end=((r_2+1)*log(2)*sqrt(N)*(log2(1+r_2)-R2_end))/sqrt(r_2^2+2*r_2);
            throughput1=N*R1_end*(1-(0.5*erfc(g_1_end/sqrt(2))));
            throughput2=N*R2_end*(1-(0.5*erfc(g_2_end/sqrt(2))));
            throughput=PrH2*throughput2+PrH1*throughput1;
            %寻找使得有效吞吐量最大的N。并记录

            if throughput>last_value
                max_throughput(j,e,eb)=throughput;%每个k和ε对应的最优的有效吞吐量
                idex_m(j,e,eb)=i;%每个k和ε对应的最优的N值
                last_value=throughput;%如果比上个值大就更新
            end

            %策略2的最优吞吐量
            pt_sol_2(i,j,e,eb)= (a2 + b2)/2;%二分法解出的pt
            if error_2==1
                pt_sol_2(i,j,e,eb)=1;
            end
            %计算策略2下的最终功率
            P1R_end_2=P1R_calc_bob(pt_sol_2(i,j,e,eb));    %H1下的接收信号功率
            P2R_end_2=P2R_calc_bob(pt_sol_2(i,j,e,eb));%H2下的接收信号功率
            %策略2下的信噪比
            r_1_2=P1R_end_2/sigmman2-1;
            r_2_2=P2R_end_2/sigmman2-1;
            %寻找0.1到1.9范围内使得有效吞吐量最大的R
            % g_1_2=@(R) ((r_1_2+1)*log(2)*sqrt(N)*(log2(1+r_1_2)-R))/sqrt(r_1_2^2+2*r_1_2);
            % g_2_2=@(R) ((r_2_2+1)*log(2)*sqrt(N)*(log2(1+r_2_2)-R))/sqrt(r_2_2^2+2*r_2_2);
            % throughput11_2=@(R) -(1-0.5*erfc(((r_1_2+1)*log(2)*sqrt(N)*(log2(1+r_1_2)-R))/sqrt(r_1_2^2+2*r_1_2)/sqrt(2)))*N*R;
            % throughput22_2=@(R) -(1-0.5*erfc(((r_2_2+1)*log(2)*sqrt(N)*(log2(1+r_2_2)-R))/sqrt(r_2_2^2+2*r_2_2)/sqrt(2)))*N*R;
            code_error1_2=@(R) 0.5*erfc((((r_1_2+1)*log(2)*sqrt(N)*(log2(1+r_1_2)-R))/sqrt(r_1_2^2+2*r_1_2))/sqrt(2))-k;
            code_error2_2=@(R) 0.5*erfc((((r_2_2+1)*log(2)*sqrt(N)*(log2(1+r_2_2)-R))/sqrt(r_2_2^2+2*r_2_2))/sqrt(2))-k;
            throughput11_2=@(R) -(1-code_error1_2(R)-k)*N*R;
            throughput22_2=@(R) -(1-code_error2_2(R)-k)*N*R;

            % [R1_opt_2,~]=fminbnd(throughput11_2,0,2);
            % [R2_opt_2,~]=fminbnd(throughput22_2,0,2);
            R1_opt_2=findmin(throughput11_2,0,0.5);
            R2_opt_2=findmin(throughput22_2,0,0.5);
            %寻找k约束下的最大R的范围，并取该值与优化后的R的最小值
            % code_error1_2=@(R) 0.5*erfc(g_1_2(R)/sqrt(2))-k(e);
            % code_error2_2=@(R) 0.5*erfc(g_2_2(R)/sqrt(2))-k(e);
            % options = optimset('TolX',0.01);
            % R1_max_2=fzero(code_error1_2,[0.1,1.2],options);
            % R1_max_2=bisection_method(code_error1_2,0,4,0.0001,e);
            % R2_max_2=bisection_method(code_error2_2,0,4,0.0001,e);%通过二分法寻找k约束下的R的最大值
            R1_max_2=findRmax(code_error1_2,0,2);
            R2_max_2=findRmax(code_error2_2,0,2);
            R1_end_2=min(R1_max_2,R1_opt_2);
            R2_end_2=min(R2_max_2,R2_opt_2);
            % if R1_max_2(i,j,e)==R1_end_2
            %     R1_max_2_indicator(i,j,e)=1;
            % end
            % if R2_max_2(i,j,e)==R1_end_2
            %     R2_max_2_indicator(i,j,e)=1;
            % end
            %得到最终的有效吞吐量
            g_1_end_2=((r_1_2+1)*log(2)*sqrt(N)*(log2(1+r_1_2)-R1_end_2))/sqrt(r_1_2^2+2*r_1_2);%Q函数内部的x
            g_2_end_2=((r_2_2+1)*log(2)*sqrt(N)*(log2(1+r_2_2)-R2_end_2))/sqrt(r_2_2^2+2*r_2_2);
            throughput1_2=N*R1_end_2*(1-(0.5*erfc(g_1_end_2/sqrt(2))));
            throughput2_2=N*R2_end_2*(1-(0.5*erfc(g_2_end_2/sqrt(2))));
            throughput_2=PrH2*throughput2_2+PrH1*throughput1_2;
            %寻找使得有效吞吐量最大的N。并记录
            if throughput_2>last_value_2
                max_throughput_2(j,e,eb)=throughput_2;%每个k和ε对应的最优的有效吞吐量
                idex_m_2(j,e,eb)=i;%每个k和ε对应的最优的N值
                last_value_2=throughput_2;%如果比上个值大就更新
            end
            % m=m+1
        end
    end
end
end

j1=1; j2=2; j3=3;                        %Q       = 0.25 / 0.5  / 0.75 在 Q 中的下标
b1=1; b2=2;                              %ibselon = 0.2  / 0.4         在 ibselon_vec 中的下标

C1_I =squeeze(max_throughput(j1,:,b1));   %Q=0.25, ibselon=0.2, strategy I  
C1_II=squeeze(max_throughput_2(j1,:,b1)); %Q=0.25, ibselon=0.2, strategy II
C2_I =squeeze(max_throughput(j2,:,b1));   %Q=0.5,  ibselon=0.2, strategy I 
C2_II=squeeze(max_throughput_2(j2,:,b1)); %Q=0.5,  ibselon=0.2, strategy II
C3_I =squeeze(max_throughput(j2,:,b2));   %Q=0.5,  ibselon=0.4, strategy I 
C3_II=squeeze(max_throughput_2(j2,:,b2)); %Q=0.5,  ibselon=0.4, strategy II
C4_I =squeeze(max_throughput(j3,:,b2));   %Q=0.75, ibselon=0.4, strategy I
C4_II=squeeze(max_throughput_2(j3,:,b2)); %Q=0.75, ibselon=0.4, strategy II

figure
hh = plot(alpha, C1_I,  'ro-',  alpha, C1_II, 'bo-', ...      %Q=0.25 ibselon=0.2 
     alpha, C2_I,  'ro--', alpha, C2_II, 'bo--', ...     %Q=0.5  ibselon=0.2
     alpha, C3_I,  'r^--', alpha, C3_II, 'b^--', ...     %Q=0.5  ibselon=0.4 
     alpha, C4_I,  'r^-.', alpha, C4_II, 'b^-.', ...     %Q=0.75 ibselon=0.4
     'LineWidth',2,'MarkerSize',7);
legend(hh([1 3 5 7]), ...
       {'$ \mathcal{Q}=0.25,\ \epsilon=0.05 $', ...
        '$ \mathcal{Q}=0.5,\ \epsilon=0.05 $', ...
        '$ \mathcal{Q}=0.5,\ \epsilon=0.25 $', ...
        '$ \mathcal{Q}=0.75,\ \epsilon=0.25 $'}, ...
        'Interpreter','latex','FontSize',12,'Location','best');
xlabel('\alpha','FontSize',12,'FontName','Times New Roman');
ylabel('C_e^T','FontSize',12,'FontName','Times New Roman');
grid on;

function taovalue = solve_tao(pt,N,PrH0,PrH1,PrH2)
rho_A = 113e-6;
sigmman2 = 0.0001;
G = 1;
delta = 0.25;
eta = 0.65;
w = 0.5;
T = [0,0];
Wil = [4,3];
d_tw = sqrt((T(1)-Wil(1))^2 + (T(2)-Wil(2))^2);
Gtw = 2;
c = 3e8;
f = 915e6;
K = c/(f*4*pi);
r = (Gtw*K^2)/(d_tw^2);

P1w = delta*pt*G;
P2w = (w*eta*pt*G - rho_A)/(1-w);

if P2w <= 0
    error('pt太小，P2w <= 0，导致功率参数无效');
end

P1R_1 = P1w*r;
P2R_1 = P2w*r;

% 原来 exp(log(a)+log(sigmman2/a+1)) 等价于 a + sigmman2
P1R = P1R_1 + sigmman2;
P2R = P2R_1 + sigmman2;

A1 = PrH1*(sigmman2/P1R)^N;
A2 = PrH2*(sigmman2/P2R)^N;

k1 = P1R_1/((sigmman2*200)*P1R);
k2 = P2R_1/((sigmman2*200)*P2R);

fRoot = @(tao) A1*exp(k1*tao) + A2*exp(k2*tao) - PrH0;

f0 = fRoot(0);

if abs(f0) < 1e-12
    taovalue1 = 0;
elseif f0 > 0
    % 根在负半轴
    lo = -1;
    while fRoot(lo) > 0 && lo > -1e12
        lo = 2*lo;
    end
    if fRoot(lo) > 0
        error('未能包围负根，请检查参数');
    end
    taovalue1 = fzero(fRoot, [lo, 0]);
else
    % 根在正半轴
    hi = 1;
    while fRoot(hi) < 0 && hi < 1e12
        hi = 2*hi;
    end
    if fRoot(hi) < 0
        error('未能包围正根，请检查参数');
    end
    taovalue1 = fzero(fRoot, [0, hi]);
end

taovalue = max(taovalue1/200,0);
end

function tao2value=solve_tao2(pt,N,PrH1,PrH2)
rho_A=113*10^(-6);             %the circuit power consumption rate (in Watt) in active transmission 113uW
sigmman2=0.0001;
G=1;                           %the energy beamforming gain
delta=0.25;                    %the power reflection coefficient
eta=0.65;                      %the efficiency of RF-to-DC energy conversion
w=0.5;                         %the time fraction for harvesting energy in active RF transmission
T=[0,0];                      %location of hybrid transmitter
Wil=[4,3];                    %location of Willie
d_tw=sqrt((T(1)-Wil(1))^2+(T(2)-Wil(2))^2);
Gtw=2;    %定向天线
c=3*10^8;
f=915*10^6;
K=c/(f*4*pi);
r=(Gtw*K^2)/(d_tw^2);
P1w=delta*pt*G;
P2w=(w*eta*pt*G-rho_A)/(1-w);
P1R_1= P1w*r;
log_sum_1=  log(P1R_1) + log(sigmman2/P1R_1+1);  % 精确计算 log(b + a)
P1R= exp(double(log_sum_1));    %H1下的接收信号功率
P2R_1= P2w*r;
log_sum_2=  log(P2R_1) + log(sigmman2/P2R_1+1);
P2R= exp(double(log_sum_2));%H2下的接收信号功率
tao2value=((P1R*P2R)/(P2R-P1R))*log(((P2R/P1R)^N)*PrH1/PrH2);
if tao2value<0
    tao2value=0;
end
% func = @(tao) PrH1*((sigmman2/P1R)^N)*exp(tao/5000*P1R_1/(sigmman2*P1R))+PrH2*((sigmman2/P2R)^N)*exp(tao/5000*P2R_1/(sigmman2*P2R))-PrH0;
% taovalue = fzero(func, [0.1, 100]);
% syms tao
% taovalue2=vpasolve(PrH1*((sigmman2/P1R)^N)*exp(tao*P1R_1/((sigmman2*200)*P1R))+PrH2*((sigmman2/P2R)^N)*exp(tao*P2R_1/((sigmman2*200)*P2R))-PrH0==0,tao);
% tao2value=taovalue2/200;
end


function P1R=P1R_calc(a)
G=1;                           %the energy beamforming gain
delta=0.25;                    %the power reflection coefficient
T=[0,0];                      %location of hybrid transmitter
Wil=[4,3];                    %location of Willie
d_tw=sqrt((T(1)-Wil(1))^2+(T(2)-Wil(2))^2);
Gtw=2;    %定向天线
sigmman2=0.0001;
c=3*10^8;
f=915*10^6;
K=c/(f*4*pi);
r=(Gtw*K^2)/(d_tw^2);
P1w_a=delta*a*G;
P1R_1_a= P1w_a*r;
log_sum_1_a=  log(P1R_1_a) + log(sigmman2/P1R_1_a+1);  % 精确计算 log(b + a)
P1R= exp(double(log_sum_1_a));    %H1下的接收信号功率
end

function P1R=P1R_calc_bob(a)
G=1;                           %the energy beamforming gain
delta=0.25;                    %the power reflection coefficient
T=[0,0];                      %location of hybrid transmitter
Bob=[2,2];                    %location of Willie
d_tr=sqrt((T(1)-Bob(1))^2+(T(2)-Bob(2))^2);
Gtw=2;    %定向天线
sigmman2=0.0001;
c=3*10^8;
f=915*10^6;
K=c/(f*4*pi);
r=(Gtw*K^2)/(d_tr^2);
P1w_a=delta*a*G;
P1R_1_a= P1w_a*r;
log_sum_1_a=  log(P1R_1_a) + log(sigmman2/P1R_1_a+1);  % 精确计算 log(b + a)
P1R= exp(double(log_sum_1_a));    %H1下的接收信号功率
end

function P2R=P2R_calc(a)
G=1;                           %the energy beamforming gain
T=[0,0];                      %location of hybrid transmitter
Wil=[4,3];                    %location of Willie
d_tw=sqrt((T(1)-Wil(1))^2+(T(2)-Wil(2))^2);
Gtw=2;    %定向天线
eta=0.65;                      %the efficiency of RF-to-DC energy conversion
w=0.5;                         %the time fraction for harvesting energy in active RF transmission
rho_A=113*10^(-6);             %the circuit power consumption rate (in Watt) in active transmission 113uW
sigmman2=0.0001;
c=3*10^8;
f=915*10^6;
K=c/(f*4*pi);
r=(Gtw*K^2)/(d_tw^2);
P2w=(w*eta*a*G-rho_A)/(1-w);
P2R_2= P2w*r;
log_sum_2=log(P2R_2) + log(sigmman2/P2R_2+1);  % 精确计算 log(b + a)
P2R= exp(double(log_sum_2));    %H2下的接收信号功率
end

function P2R=P2R_calc_bob(a)
G=1;                           %the energy beamforming gain
T=[0,0];                      %location of hybrid transmitter
Bob=[2,2];                    %location of Willie
d_tr=sqrt((T(1)-Bob(1))^2+(T(2)-Bob(2))^2);
Gtw=2;    %定向天线
eta=0.65;                      %the efficiency of RF-to-DC energy conversion
w=0.5;                         %the time fraction for harvesting energy in active RF transmission
rho_A=113*10^(-6);             %the circuit power consumption rate (in Watt) in active transmission 113uW
sigmman2=0.0001;
c=3*10^8;
f=915*10^6;
K=c/(f*4*pi);
r=(Gtw*K^2)/(d_tr^2);
P2w=(w*eta*a*G-rho_A)/(1-w);
P2R_2= P2w*r;
log_sum_2=log(P2R_2) + log(sigmman2/P2R_2+1);  % 精确计算 log(b + a)
P2R= exp(double(log_sum_2));    %H2下的接收信号功率
end

function Pe1=P_error_I(P1R,P2R,taovalue,N,ibselon,PrH0,PrH1,PrH2)
sigmman2=0.0001;
a=taovalue/sigmman2;
b1=taovalue/P1R;
b2=taovalue/P2R;
PFA=gammainc(a,N,'upper');%a到正无穷
PMD=(PrH1*gammainc(b1,N)+PrH2*gammainc(b2,N))/(PrH1+PrH2);%从0开始的
if PrH0>=0.5
    beta=1-PrH0;
else
    beta=PrH0+min(PrH1,PrH2);
end
Pe1=1/beta*(PMD*(PrH1+PrH2)+PFA*PrH0)-1+ibselon;%有无传输检错概率1
end

function Pe2=P_error_II(P1R,P2R,taovalue,taovalue2,N,PrH1,PrH2,PrH0)
% sigmman2=0.0001;
% a=taovalue/sigmman2;
% b1=taovalue/P1R;
taovalue2=max(taovalue,taovalue2);
b2=taovalue/P2R;
b1_2=taovalue2/P1R;
b2_2=taovalue2/P2R;
P_2error1=gammainc(b2,N,'upper')-gammainc(b2_2,N,'upper');
P_1error2=gammainc(b1_2,N,'upper');
if P_2error1<0
    P_2error1=0;
end
if P_1error2<0
    P_1error2=0;
end
if PrH0>=0.5
    beta=1-PrH0;
else
    beta=PrH0+min(PrH1,PrH2);
end
Pe2=1/beta*(P_2error1*PrH2+P_1error2*PrH1);

% PFA=gammainc(a,N,'upper');%a到正无穷
% PMD=(PrH1*gammainc(b1,N)+PrH2*gammainc(b2,N))/(PrH1+PrH2);%从0开始的
% Pe1=2*(PMD*(PrH1+PrH2)+PFA*PrH0)-1+ibselon;%有无传输检错概率1
end


function [Fi_1_0,Fi_2_1,Fi_0_1,Fi_1_2,Fi_0_2,Fi_2_0]=Fi_a_b_calc(P1R,P2R,N,PrH0,PrH1,PrH2)
sigmman2=0.0001;
Fi_1_0=((P1R*sigmman2)/(P1R-sigmman2))*log(((P1R/sigmman2)^N)*PrH0/PrH1);
Fi_2_1=((P2R*P1R)/(P2R-P1R))*log(((P2R/P1R)^N)*PrH1/PrH2);
Fi_0_1=((sigmman2*P1R)/(sigmman2-P1R))*log(((sigmman2/P1R)^N)*PrH1/PrH0);
Fi_1_2=((P1R*P2R)/(P1R-P2R))*log(((P1R/P2R)^N)*PrH2/PrH1);
Fi_0_2=((sigmman2*P2R)/(sigmman2-P2R))*log(((sigmman2/P2R)^N)*PrH2/PrH0);
Fi_2_0=((P2R*sigmman2)/(P2R-sigmman2))*log(((P2R/sigmman2)^N)*PrH0/PrH2);
if Fi_1_0<0
    Fi_1_0=0;
end
if Fi_2_1<0
    Fi_2_1=0;
end
if Fi_0_1<0
    Fi_0_1=0;
end
if Fi_1_2<0
    Fi_1_2=0;
end
if Fi_0_2<0
    Fi_0_2=0;
end
if Fi_2_0<0
    Fi_2_0=0;
end
end

function R_option=findmin(f,a,b)
k=(b-a)/0.001;
f_min=0;%;f_last=0;
for i=1:k
    if f_min>f(a+i*0.001)
        f_min=f(a+i*0.001);
        m=i;
    end
end
R_option=fminbnd(f,a+0.001*(m-1),a+0.001*(m+1));
end

function R_max=findRmax(f,a,b)
if abs(f((a+b)/2))<0.0001
    R_max=(a+b)/2;
elseif f((a+b)/2)<0
    R_max=fzero(f,[(a+b)/2,b]);
else
    R_max=fzero(f,[a,(a+b)/2]);
end
end
