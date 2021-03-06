%% NARX application
% Apply NARX identification method to magnetic levitation data.

%% Preapre data for identification
%[x,t] = maglev_dataset;
load('timeseries\Model_LQI_step.mat');
x=num2cell(sim_con.');
t=num2cell(1e3*sim_pos(:,2).');%;sim_vel.';sim_cur.']);
% setdemorandstream(491218381);
%%
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
plotinerrcorr(Xs,E);

%% Closed loop network
net2 = closeloop(net);
view(net2);
[Xs,Xi,Ai,Ts] = preparets(net2,x,{},t);
Y = net2(Xs,Xi,Ai);

%% Train network
[net2,tr] = train(net2,Xs,Ts,Xi,Ai);
Y = net2(Xs,Xi,Ai);

%% Response
plotresponse(Ts,Y);

%% Network without delay
net3 = removedelay(net);
view(net3);
[Xs,Xi,Ai,Ts] = preparets(net3,x,{},t);
Y = net3(Xs,Xi,Ai);

%% Train network
[net3,tr] = train(net3,Xs,Ts,Xi,Ai);
Y = net3(Xs,Xi,Ai);

%% Response
plotresponse(Ts,Y);