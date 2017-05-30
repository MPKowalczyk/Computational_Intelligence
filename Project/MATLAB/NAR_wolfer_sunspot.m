%% NAR application
% Apply NAR identification method to simple time series.

%% Preapre data for identification
load('timeseries/wolfer_sunspot.mat');
fun=@(x) cost_NAR(x,T);
[x,fval]=ga(fun,2,[],[],[],[],[5;2],[20;10],[],[1 2]);
net = narnet(1:x(2),x(1),'trainFcn','trainlm');
%view(net);
[Xs,Xi,Ai,Ts] = preparets(net,{},{},T);

%% Train the network
net = train(net,Xs,Ts,Xi,Ai);

%% Test neural network
Y = net(Xs,Xi);
perf = perform(net,Ts,Y);