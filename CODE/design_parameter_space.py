def linspace(xini,xend,xdelta=1.0):
    xini=float(xini)
    xend=float(xend)
    xdelta=float(xdelta)
    assert xini<=xend
    x=xini
    list_x=[xini]
    while x < xend:
        x+=xdelta
        list_x.append(x)
    return list_x

print '# 1.rho 2.p_shock 3.x_shock 4.alpha'
for rho in [0.0]:
    for p_shock in [0.05,1.0]:
        #list_x_shock=[round(x*0.05/p_shock,6) for x in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.1]]
        #list_x_shock=[round(x*0.05/p_shock,6) for x in [0.045, 0.95]]
        #list_x_shock=[round(x*0.05/p_shock,6) for x in [0.01, 0.1]]
        list_x_shock=[round(x*0.05/p_shock,6) for x in [0.095]]
        for x_shock in list_x_shock:
            for alpha in linspace(0.0,4.0,0.1):
            #for alpha in linspace(0.0,0.2,0.1):
                print rho, p_shock, x_shock, alpha
