function [] = ReadData(row,column,N)
%READDATA Summary of this function goes here
%   Detailed explanation goes here

columnLetter = num2xlcol(N)

ActualX=readmatrix(strcat('NewPositions',num2str(row),',',num2str(column),'.xlsx'),'Sheet','Actual Measurements x','Range',strcat('A','1',':',columnLetter,'8'));
ActualY=readmatrix(strcat('NewPositions',num2str(row),',',num2str(column),'.xlsx'),'Sheet','Actual Measurements y','Range',strcat('A','1',':',columnLetter,'8'));
ActualWidth=readmatrix(strcat('NewPositions',num2str(row),',',num2str(column),'.xlsx'),'Sheet','Actual Measurements Width','Range',strcat('A','1',':',columnLetter,'8'));
ActualHeight=readmatrix(strcat('NewPositions',num2str(row),',',num2str(column),'.xlsx'),'Sheet','Actual Measurements Height','Range',strcat('A','1',':',columnLetter,'8'));

save(strcat('ActualX',num2str(row),num2str(column),'.mat'),'ActualX');
save(strcat('ActualY',num2str(row),num2str(column),'.mat'),'ActualY');
save(strcat('ActualWidth',num2str(row),num2str(column),'.mat'),'ActualWidth');
save(strcat('ActualHeight',num2str(row),num2str(column),'.mat'),'ActualHeight');
end


function xlcol_addr=num2xlcol(col_num)
% col_num - positive integer greater than zero
    n=1;
    while col_num>26*(26^n-1)/25
        n=n+1;
    end
    base_26=zeros(1,n);
    tmp_var=-1+col_num-26*(26^(n-1)-1)/25;
    for k=1:n
        divisor=26^(n-k);
        remainder=mod(tmp_var,divisor);
        base_26(k)=65+(tmp_var-remainder)/divisor;
        tmp_var=remainder;
    end
    xlcol_addr=char(base_26); % Character vector of xlcol address
end