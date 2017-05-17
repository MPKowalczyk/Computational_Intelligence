%% Maglev Modeling
% Apply NARX identification method to magnetic levitation data.

%% Preapre data for identification
[x,t] = maglev_dataset;
% setdemorandstream(491218381);
net = narxnet(1:2,1:2,10);
view(net);
[Xs,Xi,Ai,Ts] = preparets(net,x,{},t);

%% Train the network
[net,tr] = train(net,Xs,Ts,Xi,Ai);
% nntraintool;

%% Test neural network
Y = net(Xs,Xi,Ai);
perf = mse(net,Ts,Y);
E = gsubtract(Ts,Y);

%% Figures
x_mat=cell2mat(x);
t_mat=cell2mat(t);
Y_mat=cell2mat(Y);
figure(1);
hold on;
plot(x_mat,'b');
plot(t_mat,'r');
plot(Y_mat,'g');
title('Data');
xlabel('Sample');
ylabel('Value');
legend('Control current','Position','Estimated position');
grid on;

%% Performance
plotperform(tr);

%% Response
plotresponse(Ts,Y);

%% Error autocorrelation
ploterrcorr(E);

%% Inner correlation
% plotinerrcorr(Xs,E);
% net2 = closeloop(net);
% view(net);
% [Xs,Xi,Ai,Ts] = preparets(net2,x,{},t);
% Y = net2(Xs,Xi,Ai);
% plotresponse(Ts,Y);
% net3 = removedelay(net);
% view(net);
% [Xs,Xi,Ai,Ts] = preparets(net3,x,{},t);
% Y = net3(Xs,Xi,Ai);
% plotresponse(Ts,Y);
% displayEndOfDemoMessage(mfilename);