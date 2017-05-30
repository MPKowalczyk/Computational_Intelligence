function val = cost_NAR(x,T)

val = 0;
for i = 1:4
    net = narnet(1:x(2),x(1));
    net.trainParam.showWindow=0;
    [Xs,Xi,Ai,Ts] = preparets(net,{},{},T);
    net = train(net,Xs,Ts,Xi,Ai);
    Y = net(Xs,Xi);
    val = val+perform(net,Ts,Y);
end

end