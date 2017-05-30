%% NAR application
% Apply NAR identification method to simple time series.

%% Preapre data for identification
load('timeseries/gnp_deflator.mat');
net = narnet(1:2,10,'trainFcn','trainlm');
%view(net);
[Xs,Xi,Ai,Ts] = preparets(net,{},{},T);

%% Train the network
net = train(net,Xs,Ts,Xi,Ai);

%% Test neural network
Y = net(Xs,Xi);
perf = perform(net,Ts,Y)