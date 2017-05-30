%% NAR application
% Apply NAR identification method to simple time series.

%% Preapre data for identification
T_mat = csvread('timeseries/australia_export.csv',1,1)';
T_cell = num2cell(T_mat(1:end-1));
net = narnet(1:20,40);
view(net);
[Xs,Xi,Ai,Ts] = preparets(net,{},{},T_cell);

%% Train the network
net = train(net,Xs,Ts,Xi,Ai);

%% Test neural network
Y = net(Xs,Xi);
perf = perform(net,Ts,Y);